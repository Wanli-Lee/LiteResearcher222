"""CPU-only source-mask regressions; no Ray cluster or external services.

Run: python3 training/tests/test_source_masking.py
Tool modules have Ray and filesystem side effects at import time. These tests
compile their unchanged method bodies and replace only external dependencies.
They cover tool behavior, not distributed rollout/import integration.
"""
import ast
import asyncio
import copy
from contextlib import nullcontext
import datetime
from enum import Enum
import importlib.util
import json
import logging
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from uuid import uuid4

import numpy as np

TRAINING = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("url_masking", TRAINING / "verl/tools/utils/url_masking.py")
url_masking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(url_masking)


class ToolResponse:
    def __init__(self, text="", **kwargs):
        self.text = text
        self.image = None
        self.video = None
        self.__dict__.update(kwargs)


def load_methods(relative_path, class_name, names):
    path = TRAINING / relative_path
    source = ast.parse(path.read_text())
    original = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    methods = [n for n in original.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    if {n.name for n in methods} != set(names):
        raise AssertionError("A tested production method is missing")
    supporting_classes = [n for n in source.body if isinstance(n, ast.ClassDef) and n.name in {"AgentState", "AgentData"}]
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *supporting_classes,
        ast.ClassDef(name=class_name, bases=[], keywords=[], body=methods, decorator_list=[]),
    ], type_ignores=[])
    logger = logging.getLogger("source-mask-test")
    scope = dict(re=re, json=json, datetime=datetime.datetime, uuid4=uuid4,
                 asyncio=asyncio, copy=copy, Enum=Enum, np=np, AgentLoopOutput=SimpleNamespace,
                 rollout_trace_op=lambda fn: fn, simple_timer=lambda *args: nullcontext(),
                 logger=logger, _url_mask_logger=logger, ToolResponse=ToolResponse,
                 log_url_mask=lambda *args: None,
                 clean_mask_url=url_masking.clean_mask_url, url_matches_mask=url_masking.url_matches_mask,
                 bind_source_mask=url_masking.bind_source_mask)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[class_name]


Search = load_methods("verl/tools/google_search_tool.py", "GoogleSearchTool", {
    "create", "execute", "execute_search", "release", "_parse_search_result_block",
    "_deduplicate_per_query", "_format_results_per_query", "_mask_url_in_search_result",
})
Browse = load_methods("verl/tools/browse_tool.py", "BrowseTool", {
    "create", "execute", "execute_browse", "release",
})
Loop = load_methods("verl/experimental/agent_loop/tool_agent_loop.py", "ToolAgentLoop", {
    "run", "_handle_pending_state", "_handle_generating_state", "_handle_processing_tools_state", "_call_tool",
})
Manager = load_methods("verl/experimental/agent_loop/agent_loop.py", "AgentLoopManager", {"_performance_metrics"})


class Pool:
    def __init__(self, response):
        self.calls = []
        self.response = response
        self.execute = SimpleNamespace(remote=self.call)

    async def call(self, fn, *args):
        self.calls.append(args)
        return self.response


def search_tool(blocks, dedup):
    tool = Search()
    tool._instance_dict = {}
    tool.enable_dedup = dedup
    tool.search_service_url = "unused"
    tool.default_limit = 10
    tool.timeout = 1
    results = [(json.dumps({"result": block}), {}) for block in blocks]
    tool.execution_pool = Pool((results, {"status": "success"}))
    return tool


def browse_tool():
    tool = Browse()
    tool._instance_dict = {}
    tool.browse_service_url = "unused"
    tool.timeout = 1
    tool.strip_evidence = False
    tool.url_check_service_url = ""
    tool.execution_pool = Pool(("ALLOWED_PAGE_CONTENT", {"status": "success", "from_cache": True}))
    return tool


def block(urls):
    return "A Google search found results:\n\n## Web Results\n" + "\n\n".join(
        f"{i}. [Page {i}]({url})\n{snippet}" for i, (url, snippet) in enumerate(urls, 1)
    ) + "\n\n"


class MatchingTests(unittest.TestCase):
    def test_mask_counts_reach_global_rollout_metrics(self):
        metrics = [[{"generate_sequences": 1, "tool_calls": 1, "tool/search/masked_result_count": 2,
                     "tool/search/mask_configured_calls": 1, "tool/search/mask_hit_calls": 1}],
                   [{"generate_sequences": 2, "tool_calls": 1, "tool/visit/masked_url_count": 1,
                     "tool/visit/mask_configured_calls": 2, "tool/visit/mask_hit_calls": 1}]]
        output = SimpleNamespace(batch={"attention_mask": np.array([[1, 1, 1], [1, 1, 0]]),
                                        "prompts": np.array([[1], [1]])})
        merged = Manager()._performance_metrics(metrics, output)
        self.assertEqual(merged["tool/search/masked_result_count"], 2)
        self.assertEqual(merged["tool/visit/masked_url_count"], 1)
        self.assertEqual(merged["tool/visit/mask_configured_calls"], 2)
        self.assertEqual(merged["tool/visit/mask_hit_calls"], 1)

    def test_legacy_parenthesis_and_legitimate_title(self):
        source = "https://example.org/wiki/Title_(part_(one))"
        self.assertEqual(url_masking.clean_mask_url(source), source)
        self.assertEqual(url_masking.clean_mask_url(source + "))"), source)
        self.assertTrue(url_masking.url_matches_mask(source, source + ")"))

    def test_fragments_and_host_case(self):
        self.assertTrue(url_masking.url_matches_mask("https://EXAMPLE.org/page#section", "https://example.org/page"))

    def test_distinct_resources_are_not_masked(self):
        for candidate in ["https://example.org/page2", "https://other.org/page", "https://example.org/PAGE",
                          "https://example.org/page?id=2", "https://other.org/?next=https://example.org/page"]:
            with self.subTest(candidate=candidate):
                self.assertFalse(url_masking.url_matches_mask(candidate, "https://example.org/page"))
        self.assertFalse(url_masking.url_matches_mask("https://example.org/?id=2", "https://example.org/?id=1"))

    def test_empty_mask_does_not_filter(self):
        for empty in [None, "", " ", 0]:
            self.assertFalse(url_masking.url_matches_mask("https://example.org/page", empty))

    def test_bind_source_to_both_tools_without_mutating_sample(self):
        extra = {"mask_url": "https://example.org/source)", "tools_kwargs": {"browse": {"timeout": 5}}}
        original = copy.deepcopy(extra)
        resolved = url_masking.bind_source_mask(None, extra)
        for name in ("search", "visit"):
            self.assertEqual(resolved[name]["create_kwargs"]["url"], "https://example.org/source")
        self.assertEqual(resolved["visit"]["timeout"], 5)
        self.assertEqual(extra, original)

    def test_metadata_source_cannot_be_disabled_by_empty_tool_kwargs(self):
        resolved = url_masking.bind_source_mask(
            {"visit": {}, "search": {"create_kwargs": {"url": "https://example.org/wrong"}}},
            {"mask_url": "https://example.org/source)"},
        )
        for name in ("search", "visit"):
            self.assertEqual(resolved[name]["create_kwargs"]["url"], "https://example.org/source")

    def test_legacy_sources_are_recovered(self):
        for name in ("search", "visit", "browse"):
            resolved = url_masking.bind_source_mask({name: {"create_kwargs": {"url": "https://example.org/source)"}}})
            for target in ("search", "visit"):
                self.assertEqual(resolved[target]["create_kwargs"]["url"], "https://example.org/source")
        resolved = url_masking.bind_source_mask({}, {"url": "https://example.org/source)"})
        self.assertEqual(resolved["search"]["create_kwargs"]["url"], "https://example.org/source")


class SearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_bracketed_page_title_does_not_bypass_mask(self):
        response_block = "1. [Source [PDF]](https://example.org/source)\r\nSOURCE_SECRET\r\n\r\n"
        response_block += "2. [Other [HTML]](https://example.org/other)\r\nSAFE\r\n\r\n"
        for dedup in (True, False):
            tool = search_tool([response_block], dedup)
            instance, _ = await tool.create(create_kwargs={"url": "https://example.org/source)"})
            response, _, metrics = await tool.execute(instance, {"query": ["test"]})
            self.assertNotIn("SOURCE_SECRET", response.text)
            self.assertIn("SAFE", response.text)
            self.assertEqual(metrics["masked_result_count"], 1)

    async def test_source_removed_and_neighbor_snippets_preserved(self):
        # First/middle/last entries, both dedup paths, ordinary and nested URLs.
        for dedup in [True, False]:
            for source in ["https://example.org/source", "https://example.org/wiki/Title_(part_(one))"]:
                for position in range(3):
                    with self.subTest(dedup=dedup, source=source, position=position):
                        entries = [("https://example.org/other1", "NEIGHBOR_ONE"), ("https://example.org/other2", "NEIGHBOR_TWO")]
                        entries.insert(position, (source, "SOURCE_SECRET"))
                        tool = search_tool([block(entries)], dedup)
                        instance, _ = await tool.create(create_kwargs={"url": source + ")"})
                        response, _, metrics = await tool.execute(instance, {"query": ["test"]})
                        self.assertNotIn("SOURCE_SECRET", response.text)
                        self.assertNotIn(source, response.text)
                        self.assertIn("NEIGHBOR_ONE", response.text)
                        self.assertIn("NEIGHBOR_TWO", response.text)
                        self.assertTrue(metrics["url_mask_configured"])
                        self.assertTrue(metrics["url_masked"])
                        self.assertEqual(metrics["masked_result_count"], 1)
                        self.assertEqual(metrics["original_result_count"], 3)
                        self.assertEqual(metrics["deduped_result_count"], 2)

    async def test_configured_mask_without_match_is_not_reported_as_removed(self):
        for dedup in [True, False]:
            tool = search_tool([block([("https://example.org/other", "SAFE")])], dedup)
            instance, _ = await tool.create(create_kwargs={"url": "https://example.org/source)"})
            response, _, metrics = await tool.execute(instance, {"query": ["test"]})
            self.assertIn("SAFE", response.text)
            self.assertTrue(metrics["url_mask_configured"])
            self.assertFalse(metrics["url_masked"])
            self.assertEqual(metrics["masked_result_count"], 0)

    async def test_empty_mask_preserves_results(self):
        for dedup in [True, False]:
            tool = search_tool([block([("https://example.org/source", "SOURCE_VISIBLE")])], dedup)
            instance, _ = await tool.create()
            response, _, metrics = await tool.execute(instance, {"query": ["test"]})
            self.assertIn("SOURCE_VISIBLE", response.text)
            self.assertFalse(metrics["url_mask_configured"])
            self.assertFalse(metrics["url_masked"])

    async def test_multiple_queries_and_duplicate_counts(self):
        source = ("https://example.org/source", "SOURCE_SECRET")
        other = ("https://example.org/other", "SAFE")
        for dedup, expected_count in [(True, 1), (False, 2)]:
            tool = search_tool([block([source, other]), block([other, source])], dedup)
            instance, _ = await tool.create(create_kwargs={"url": source[0] + ")"})
            response, _, metrics = await tool.execute(instance, {"query": ["first query", "second query"]})
            self.assertNotIn("SOURCE_SECRET", response.text)
            self.assertEqual(response.text.count("SAFE"), expected_count)
            self.assertIn("first query", response.text)
            self.assertIn("second query", response.text)
            self.assertEqual(metrics["masked_result_count"], 2)
            self.assertEqual(metrics["original_result_count"], 4)
            self.assertEqual(metrics["deduped_result_count"], expected_count)

    async def test_all_search_results_masked(self):
        for dedup in [True, False]:
            tool = search_tool([block([("https://example.org/source", "SOURCE_SECRET")])], dedup)
            instance, _ = await tool.create(create_kwargs={"url": "https://example.org/source)"})
            response, _, metrics = await tool.execute(instance, {"query": ["test"]})
            self.assertNotIn("SOURCE_SECRET", response.text)
            self.assertIn("0 results", response.text)
            self.assertEqual(metrics["deduped_result_count"], 0)


class BrowseTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_source_visit_never_reaches_backend(self):
        tool = browse_tool()
        instance, _ = await tool.create(create_kwargs={"url": "https://example.org/source)"})
        response, _, metrics = await tool.execute(instance, {"url": "https://example.org/source#section", "goal": "find"})
        self.assertEqual(tool.execution_pool.calls, [])
        self.assertNotIn("ALLOWED_PAGE_CONTENT", response.text)
        self.assertNotIn("https://example.org/source", response.text)
        self.assertEqual(metrics["masked_url_count"], 1)
        self.assertEqual(metrics["fetched_url_count"], 0)
        self.assertEqual(metrics["status"], "source_masked")

    async def test_mixed_visit_only_fetches_allowed_page(self):
        tool = browse_tool()
        instance, _ = await tool.create(create_kwargs={"url": "https://example.org/source)"})
        response, _, metrics = await tool.execute(instance, {"url": ["https://example.org/source", "https://example.org/other"], "goal": "find"})
        self.assertEqual([args[1] for args in tool.execution_pool.calls], ["https://example.org/other"])
        self.assertIn("ALLOWED_PAGE_CONTENT", response.text)
        self.assertEqual(metrics["url_count"], 2)
        self.assertEqual(metrics["masked_url_count"], 1)
        self.assertEqual(metrics["fetched_url_count"], 1)
        self.assertEqual(metrics["cache_hit_count"], 1)

    async def test_mask_is_per_instance(self):
        tool = browse_tool()
        blocked, _ = await tool.create(create_kwargs={"url": "https://example.org/source)"})
        allowed, _ = await tool.create()
        outputs = await asyncio.gather(*[
            tool.execute(instance, {"url": "https://example.org/source", "goal": "find"}) for instance in [blocked, allowed]
        ])
        self.assertEqual(len(tool.execution_pool.calls), 1)
        self.assertTrue(outputs[0][2]["url_masked"])
        self.assertFalse(outputs[1][2]["url_masked"])

    async def test_legacy_browse_kwargs_reach_visit_and_release_instance(self):
        tool = browse_tool()
        loop = Loop()
        loop.tools = {"visit": tool}
        loop.max_tool_response_length = 10000
        loop.tool_response_truncate_side = "right"
        call = SimpleNamespace(name="visit", arguments=json.dumps({"url": "https://example.org/source", "goal": "find"}))
        response, _, metrics = await loop._call_tool(call, {"browse": {"create_kwargs": {"url": "https://example.org/source)"}}})
        self.assertEqual(tool.execution_pool.calls, [])
        self.assertTrue(metrics["url_masked"])
        self.assertEqual(tool._instance_dict, {})

    async def test_explicit_visit_kwargs_override_legacy_alias(self):
        tool = browse_tool()
        loop = Loop()
        loop.tools = {"visit": tool}
        loop.max_tool_response_length = 10000
        loop.tool_response_truncate_side = "right"
        call = SimpleNamespace(name="visit", arguments=json.dumps({"url": "https://example.org/source", "goal": "find"}))
        response, _, metrics = await loop._call_tool(call, {"visit": {}, "browse": {"create_kwargs": {"url": "https://example.org/source)"}}})
        self.assertEqual(len(tool.execution_pool.calls), 1)
        self.assertFalse(metrics["url_mask_configured"])
        self.assertEqual(tool._instance_dict, {})




class Tokenizer:
    def __init__(self):
        self.tool_messages = []

    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, ids, **kwargs):
        return bytes(ids).decode()

    def apply_chat_template(self, messages, **kwargs):
        self.tool_messages.extend(m["content"] for m in messages if m["role"] == "tool")
        return self.encode("\n".join(m["content"] for m in messages))


class ScriptedModel:
    """Only model generation and tool-call parsing are simulated."""
    def __init__(self, tokenizer, source):
        self.tokenizer = tokenizer
        self.turn = 0
        self.contexts = []
        self.replies = [
            json.dumps({"name": "search", "arguments": {"query": ["find evidence"]}}),
            json.dumps({"name": "visit", "arguments": {"url": [source, "https://example.org/other"], "goal": "verify"}}),
            "<answer>done</answer>",
        ]

    async def generate(self, **kwargs):
        self.contexts.append(self.tokenizer.decode(kwargs["prompt_ids"]))
        reply = self.replies[self.turn]
        self.turn += 1
        return SimpleNamespace(token_ids=self.tokenizer.encode(reply), log_probs=None, finish_reason="stop")

    async def extract_tool_calls(self, ids):
        data = json.loads(self.tokenizer.decode(ids))
        return "", [SimpleNamespace(name=data["name"], arguments=json.dumps(data["arguments"]))]


def full_loop(source):
    loop = Loop()
    loop.loop = asyncio.get_running_loop()
    loop.tokenizer = Tokenizer()
    loop.server_manager = ScriptedModel(loop.tokenizer, source)
    loop.tool_parser = loop.server_manager
    loop.processor = None
    loop.apply_chat_template_kwargs = {}
    loop.interaction_config_file = None
    loop.response_length = 50000
    loop.max_assistant_turns = 8
    loop.max_user_turns = 8
    loop.max_parallel_calls = 4
    loop.max_tool_response_length = 10000
    loop.tool_response_truncate_side = "right"
    loop.system_prompt = []
    loop.terminate_on_answer = True
    loop.tools = {
        "search": search_tool([block([(source, "SOURCE_SECRET"), ("https://example.org/other", "SAFE_SEARCH_CONTENT")])], True),
        "visit": browse_tool(),
    }
    return loop


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_masks_source_in_search_visit_and_next_model_context(self):
        source = "https://example.org/wiki/Source_(page)"
        legacy = {"create_kwargs": {"url": source + ")"}}
        cases = [
            {"extra_info": {"mask_url": source + ")"}},
            {"extra_info": {"url": source + ")"}},
            {"tools_kwargs": {"browse": legacy}},
            {"extra_info": {"mask_url": source + ")"}, "tools_kwargs": {"search": {}, "visit": {}}},
            {"extra_info": {"mask_url": source + ")", "tools_kwargs": {"search": legacy, "browse": legacy}}},
        ]
        for case in cases:
            with self.subTest(case=case):
                loop = full_loop(source)
                original = copy.deepcopy(case)
                output = await loop.run({}, raw_prompt=[{"role": "user", "content": "Find the fact."}], **case)
                self.assertEqual(case, original)
                tool_text = "\n".join(loop.tokenizer.tool_messages)
                self.assertNotIn("SOURCE_SECRET", tool_text)
                self.assertNotIn("SOURCE_SECRET", "\n".join(loop.server_manager.contexts))
                self.assertIn("SAFE_SEARCH_CONTENT", tool_text)
                self.assertIn("ALLOWED_PAGE_CONTENT", tool_text)
                self.assertEqual([args[1] for args in loop.tools["visit"].execution_pool.calls], ["https://example.org/other"])
                self.assertEqual(output.metrics["tool/search/masked_result_count"], 1)
                self.assertEqual(output.metrics["tool/visit/masked_url_count"], 1)
                for name in ("search", "visit"):
                    self.assertEqual(output.metrics[f"tool/{name}/mask_configured_calls"], 1)
                    self.assertEqual(output.metrics[f"tool/{name}/mask_hit_calls"], 1)
                    self.assertEqual(loop.tools[name]._instance_dict, {})
                self.assertIn("<answer>done</answer>", loop.tokenizer.decode(output.response_ids))

    async def test_run_without_source_does_not_suppress_content(self):
        loop = full_loop("https://example.org/source")
        output = await loop.run({}, raw_prompt=[{"role": "user", "content": "Find the fact."}], extra_info={"mask_url": ""})
        self.assertIn("SOURCE_SECRET", "\n".join(loop.tokenizer.tool_messages))
        self.assertEqual(len(loop.tools["visit"].execution_pool.calls), 2)
        self.assertEqual(output.metrics["tool/search/masked_result_count"], 0)
        self.assertEqual(output.metrics["tool/visit/masked_url_count"], 0)
        self.assertEqual(output.metrics["tool/search/mask_configured_calls"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
