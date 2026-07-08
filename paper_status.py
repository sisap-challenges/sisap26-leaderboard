from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).parent
TEAMS_JSON = SCRIPT_DIR / "data" / "teams.json"


@lru_cache(maxsize=1)
def load_paper_statuses() -> dict[str, str]:
    if not TEAMS_JSON.exists():
        return {}

    data = yaml.safe_load(TEAMS_JSON.read_text()) or {}
    if not isinstance(data, dict):
        return {}

    statuses: dict[str, str] = {}
    for team, meta in data.items():
        if not isinstance(team, str):
            continue
        if isinstance(meta, dict) and meta.get("paper") is not None:
            statuses[team] = "under review"
        else:
            statuses[team] = "---"
    return statuses


def get_paper_status(*team_keys: str) -> str:
    statuses = load_paper_statuses()
    for key in team_keys:
        if key and key in statuses:
            return statuses[key]
    return "---"
