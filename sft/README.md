# SFT Cold-Start

Full-parameter SFT of `Qwen3-4B-Thinking-2507` on the 68.2 K distilled
deep-research trajectories, producing the checkpoint that Stage-1 RL starts from.

This is the **pre-step** of the pipeline:

```
SFT cold-start (here)  →  Stage-1 RL  →  Stage-2 RL  →  71.3% GAIA / 78.0% Xbench-DS
```

> **You probably don't need to run this.** The resulting checkpoint is already
> released as
> [`simplex-ai-inc/LiteResearcher-4B-SFT`](https://huggingface.co/simplex-ai-inc/LiteResearcher-4B-SFT),
> and [`training/`](../training/) pulls it automatically. Run this only if you
> want to reproduce the cold-start, change the base model, or train on your own
> trajectory mix.

---

## Do I need to preprocess the data?

**No.** The released dataset is already in the exact format LLaMA-Factory
consumes — ShareGPT-style `conversations`, tool observations folded into `user`
turns, system prompt as the first message. There is no conversion, no
tokenization, and no filtering step to run.

`dataset_info.json` here ships two entries; pick one:

**Option A — stream from the Hub (default, nothing to download manually)**

`qwen3_4b_sft.yaml` already sets `dataset: literesearcher_sft`, which resolves to
`hf_hub_url: simplex-ai-inc/LiteResearcher-SFT-Data`. Just run `train.sh`; the
`datasets` library fetches and caches the parquet shards on first use.

**Option B — download once, train offline**

```bash
hf download simplex-ai-inc/LiteResearcher-SFT-Data --repo-type dataset \
            --local-dir ./literesearcher_sft_data     # 68,231 rows, 732 MB
```

Then switch one line in `qwen3_4b_sft.yaml`:

```yaml
dataset: literesearcher_sft_local
```

> ⚠️ The `literesearcher_sft_local` entry points at `literesearcher_sft_data/data`,
> **not** at the repo root. LLaMA-Factory loads a directory by listing every file
> in it and requires them to share one extension — the repo root also contains
> `README.md` and `.gitattributes`, which would make it error out. The `data/`
> subdirectory holds only the nine parquet shards.

---

## What the trainer does with the data

Worth knowing before you read the logs:

- **Loss masking** — LLaMA-Factory trains on `assistant` turns and masks
  everything else. Since search results and fetched pages live in `user` turns,
  tool output is masked out automatically. No extra config needed.
- **The `observation_tag` in `dataset_info.json` is inert.** It's kept so the
  entry matches the LLaMA-Factory schema, but no message in this dataset carries
  the `observation` role.
- **8 of 68,231 rows are dropped.** They have two consecutive `assistant` turns
  at the tail, which fails LLaMA-Factory's strict user/assistant alternation
  check. You will see `Invalid role tag at turn ...` warnings — this is expected,
  and leaves **68,223** trainable rows.
- **533 optimizer steps = 1 epoch** (68,223 ÷ global batch 128). The released
  checkpoint is the step-533 snapshot.

---

## Requirements

- **GPU** — 8 × H100 80 GB, single node. 64 K context with full-parameter
  training needs gradient checkpointing + Liger kernel; both are on in the yaml.
- **Software**

```bash
pip install "llamafactory[torch,metrics,deepspeed,liger-kernel]"
pip install flash-attn --no-build-isolation
```

Verified against LLaMA-Factory `0.9.4`. The `enable_thinking` and
`neat_packing` keys require ≥ 0.9.2.

---

## Run

```bash
cd sft
export WANDB_API_KEY=...            # optional; unset -> offline logging
export WANDB_PROJECT=literesearcher-sft
bash train.sh
```

Checkpoints land in `sft/saves/literesearcher-4b-sft/`, one every 50 steps
(`save_only_model: true`, so no optimizer state is kept — you cannot resume, but
each checkpoint is directly loadable).

To use the result for Stage-1 RL:

```bash
export MODEL_PATH=/path/to/sft/saves/literesearcher-4b-sft/checkpoint-533
```

---

## Files

| File | What it is |
|---|---|
| `qwen3_4b_sft.yaml` | LLaMA-Factory training config — the hyper-parameters we used |
| `dataset_info.json` | Dataset registration; Hub-streaming and local-directory variants |
| `ds_z2_config.json` | DeepSpeed ZeRO-2 config |
| `train.sh` | Launcher — sets env, runs `llamafactory-cli train`, tees a log |

## Hyper-parameters

| | |
|---|---|
| Base model | `Qwen/Qwen3-4B-Thinking-2507` |
| Method | Full-parameter SFT, DeepSpeed ZeRO-2 |
| Max length | 64 K (`cutoff_len: 65536`) |
| Global batch | 128 (per-device 2 × grad-accum 8 × 8 GPUs) |
| LR / schedule | 2e-5, cosine, 10 % warmup |
| Weight decay | 0.01 |
| Epochs | 1 |
| Precision | bf16 |
| Seed | 42 (`seed` and `data_seed`) |

We pick the Thinking-2507 base so the policy already carries long
chain-of-thought (`<think>...</think>`) behaviour into the RL stage. A SFT
cold-start is a hard requirement — pure RL from a vanilla Qwen3-4B does not learn
to use the tools reliably enough to bootstrap rewards.

---

## Links

- Dataset — [🤗 `LiteResearcher-SFT-Data`](https://huggingface.co/datasets/simplex-ai-inc/LiteResearcher-SFT-Data)
- Resulting checkpoint — [🤗 `LiteResearcher-4B-SFT`](https://huggingface.co/simplex-ai-inc/LiteResearcher-4B-SFT)
- Next step — [`training/`](../training/) (Stage-1 / Stage-2 RL)
- Paper — [arXiv:2604.17931](https://arxiv.org/abs/2604.17931)
