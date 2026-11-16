import base64
import os

import httpx
from langchain_core.tools import tool

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "")


@tool
def get_repo_info(owner: str, repo: str) -> str:
    """Get basic info about a GitHub repository (description, language, stars, URL)."""
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    response = httpx.get(f"https://api.github.com/repos/{owner}/{repo}", headers=headers, timeout=15)
    if response.status_code != 200:
        return f"GitHub error {response.status_code}: {response.text[:200]}"
    data = response.json()
    return (
        f"{data['full_name']}\n"
        f"Description: {data.get('description')}\n"
        f"Language: {data.get('language')}\n"
        f"Stars: {data.get('stargazers_count')}\n"
        f"URL: {data['html_url']}"
    )


@tool
def get_repo_issues(owner: str, repo: str, limit: int = 10) -> str:
    """List open issues for a GitHub repository. Returns titles and URLs."""
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    response = httpx.get(
        f"https://api.github.com/repos/{owner}/{repo}/issues",
        params={"state": "open", "per_page": limit},
        headers=headers,
        timeout=15,
    )
    if response.status_code != 200:
        return f"GitHub error {response.status_code}: {response.text[:200]}"
    issues = [i for i in response.json() if "pull_request" not in i]  # API mixes in PRs
    if not issues:
        return "No open issues."
    return "\n".join(f"#{i['number']} {i['title']} - {i['html_url']}" for i in issues)


def _gh_get(endpoint: str, params: dict | None = None) -> httpx.Response:
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return httpx.get(f"https://api.github.com{endpoint}", params=params, headers=headers, timeout=15)


@tool
def get_pull_requests(owner: str, repo: str, state: str = "open", limit: int = 10) -> str:
    """List pull requests in a GitHub repository. state can be 'open', 'closed', or 'all'."""
    response = _gh_get(f"/repos/{owner}/{repo}/pulls", {"state": state, "per_page": limit})
    if response.status_code != 200:
        return f"GitHub error {response.status_code}: {response.text[:200]}"
    prs = response.json()
    if not prs:
        return f"No {state} pull requests."
    return "\n".join(f"#{p['number']} {p['title']} ({p['state']}) - {p['html_url']}" for p in prs)


@tool
def get_repo_files(owner: str, repo: str, path: str = "") -> str:
    """List files and folders at a path in a GitHub repository. Leave path empty for the top level."""
    response = _gh_get(f"/repos/{owner}/{repo}/contents/{path}")
    if response.status_code != 200:
        return f"GitHub error {response.status_code}: {response.text[:200]}"
    items = response.json()
    if isinstance(items, dict):
        return f"'{path}' is a file, not a folder. Use read_repo_file to read it."
    return "\n".join(f"{item['path']}{'/' if item['type'] == 'dir' else ''}" for item in items)


@tool
def read_repo_file(owner: str, repo: str, path: str) -> str:
    """Read the contents of one file in a GitHub repository, e.g. 'README.md'."""
    response = _gh_get(f"/repos/{owner}/{repo}/contents/{path}")
    if response.status_code != 200:
        return f"GitHub error {response.status_code}: {response.text[:200]}"
    data = response.json()
    if isinstance(data, list):
        return f"'{path}' is a folder. Use get_repo_files to list it."
    text = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
    return f"Source: {data['html_url']}\n\n{text[:8000]}"  # cap length to keep the model's input small


@tool
def list_repos(username: str = "", limit: int = 30) -> str:
    """List GitHub repositories for a user. Leave username empty to list the owner's own repos (defaults to GITHUB_USERNAME in .env)."""
    username = username or GITHUB_USERNAME
    if username:
        endpoint = f"/users/{username}/repos"
    elif GITHUB_TOKEN:
        endpoint = "/user/repos"  # includes private repos of the token owner
    else:
        return "Give a username, or set GITHUB_TOKEN to list your own repos."
    response = _gh_get(endpoint, {"sort": "updated", "per_page": min(limit, 100)})
    if response.status_code != 200:
        return f"GitHub error {response.status_code}: {response.text[:200]}"
    repos = response.json()
    if not repos:
        return "No repositories found."
    lines = [
        f"{r['full_name']} ({'private' if r['private'] else 'public'}, {r.get('language') or 'no language'}, "
        f"{r['stargazers_count']} stars) - {r['html_url']}"
        for r in repos
    ]
    return "\n".join(lines)


GITHUB_TOOLS = [get_repo_info, get_repo_issues, get_pull_requests, get_repo_files, read_repo_file, list_repos]
ALL_TOOLS = GITHUB_TOOLS
