from __future__ import annotations

import json
from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PRIVATE_REPO_LABEL = "Private"


def normalize_github_repo(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""

    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.removeprefix("git@github.com:")

    if url.endswith(".git"):
        url = url[:-4]

    if "/tree/" in url:
        url = url.split("/tree/", 1)[0]

    if "/commit/" in url:
        url = url.split("/commit/", 1)[0]

    if url.endswith("/"):
        url = url[:-1]

    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"} and parsed.netloc == "github.com":
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            return f"https://github.com/{parts[0]}/{parts[1]}"
    return url


@lru_cache(maxsize=None)
def is_public_github_repo(url: str) -> bool:
    normalized_url = normalize_github_repo(url)
    if not normalized_url:
        return False

    parsed = urlparse(normalized_url)
    if parsed.scheme not in {"http", "https"} or parsed.netloc != "github.com":
        return True

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return False

    api_url = f"https://api.github.com/repos/{parts[0]}/{parts[1]}"
    request = Request(
        api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "sisap26-website-importer",
        },
    )

    try:
        with urlopen(request, timeout=10) as response:
            payload = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            return False
        if exc.code == 403:
            return is_public_github_repo_via_web(normalized_url)
        raise RuntimeError(f"GitHub API request failed for {normalized_url}: HTTP {exc.code}") from exc
    except URLError:
        return is_public_github_repo_via_web(normalized_url)

    return not bool(payload.get("private", False))


@lru_cache(maxsize=None)
def is_public_github_repo_via_web(url: str) -> bool:
    request = Request(
        url,
        headers={
            "User-Agent": "sisap26-website-importer",
        },
    )

    try:
        with urlopen(request, timeout=10) as response:
            return response.geturl().rstrip("/") == url.rstrip("/")
    except HTTPError as exc:
        if exc.code == 404:
            return False
        raise RuntimeError(f"GitHub web request failed for {url}: HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"GitHub web request failed for {url}: {exc.reason}") from exc


def public_repo_or_private_label(url: str) -> str:
    normalized_url = normalize_github_repo(url)
    if not normalized_url:
        return ""
    return normalized_url if is_public_github_repo(normalized_url) else PRIVATE_REPO_LABEL
