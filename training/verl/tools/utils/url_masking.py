"""Conservative matching for per-sample source URLs in search results."""

from copy import deepcopy
from urllib.parse import urlsplit, urlunsplit


def clean_mask_url(url: str) -> str:
    """Repair extra Markdown closing parentheses in legacy mask metadata.

    Balanced parentheses (e.g. Wikipedia titles) and URL query parameters
    are significant and must be preserved. No input dataset is rewritten.
    """
    if not isinstance(url, str):
        return ""
    url = url.strip()
    while url.endswith(")") and url.count(")") > url.count("("):
        url = url[:-1]
    return url


def _document_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    url = url.strip()
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        return url
    # Fragments identify locations within a page. Paths and queries remain
    # case-sensitive and are not decoded, truncated, or stripped of parameters.
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query, ""))


def url_matches_mask(url: str, masked_url: str) -> bool:
    """Match a source page, without masking unrelated URL prefixes/substrings."""
    source = _document_url(clean_mask_url(masked_url))
    return bool(source) and _document_url(url) == source


def bind_source_mask(tools_kwargs: dict, extra_info: dict = None) -> dict:
    """Bind the sample's source to search without changing the row or visit kwargs.

    The dataset's mask_url (or legacy url) is authoritative. If it is absent,
    recover the source from existing search/visit/browse construction kwargs.
    A known source must not disappear because search kwargs are missing.
    Source masking applies only to search; visit remains unrestricted.
    """
    extra_info = extra_info or {}
    resolved = deepcopy(extra_info.get("tools_kwargs") or {})
    resolved.update(deepcopy(tools_kwargs or {}))
    source = clean_mask_url(extra_info.get("mask_url") or extra_info.get("url"))
    for name in ("search", "visit", "browse"):
        if source:
            break
        create_kwargs = (resolved.get(name) or {}).get("create_kwargs") or {}
        source = clean_mask_url(create_kwargs.get("url"))
    if not source:
        return resolved

    tool_kwargs = deepcopy(resolved.get("search") or {})
    create_kwargs = dict(tool_kwargs.get("create_kwargs") or {})
    create_kwargs["url"] = source
    tool_kwargs["create_kwargs"] = create_kwargs
    resolved["search"] = tool_kwargs
    return resolved
