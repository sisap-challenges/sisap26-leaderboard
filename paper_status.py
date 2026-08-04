from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).parent
TEAMS_JSON = SCRIPT_DIR / "data" / "teams.json"


@lru_cache(maxsize=1)
def load_teams() -> dict:
    if not TEAMS_JSON.exists():
        return {}
    data = yaml.safe_load(TEAMS_JSON.read_text()) or {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def load_aliases() -> dict[str, str]:
    """Map every alias of a merged team onto its canonical team name.

    A team that registered more than once (e.g. under one name per submitted
    system) lists its registration names under "aliases"; the leaderboard then
    reports all of its runs under the single canonical name.
    """
    aliases: dict[str, str] = {}
    for team, meta in load_teams().items():
        if not isinstance(team, str) or not isinstance(meta, dict):
            continue
        for alias in meta.get("aliases") or []:
            aliases[alias] = team
    return aliases


def resolve_team(team_key: str) -> str:
    """Canonical team name for a registration name."""
    return load_aliases().get(team_key, team_key)


def is_scored(team_key: str) -> bool:
    """False for the surplus entries of a merged team.

    A team may enter each task once, so a merged team nominates one of its
    registrations as the scored one via "scored-alias". Its other runs are
    reported out of competition on the archived leaderboard.
    """
    canonical = resolve_team(team_key)
    meta = load_teams().get(canonical)
    if not isinstance(meta, dict):
        return True
    scored = meta.get("scored-alias")
    if not scored or team_key == canonical:
        return True
    return team_key == scored


@lru_cache(maxsize=1)
def load_paper_statuses() -> dict[str, str]:
    statuses: dict[str, str] = {}
    for team, meta in load_teams().items():
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
        if not key:
            continue
        canonical = resolve_team(key)
        if canonical in statuses:
            return statuses[canonical]
    return "---"


def has_paper(*team_keys: str) -> bool:
    """True if the team contributed a system-description paper.

    This is what separates the two leaderboard views: the public leaderboard
    shows every run, while the archived SISAP leaderboard is restricted to
    entries backed by a paper (plus the organizers' baselines).
    """
    return get_paper_status(*team_keys) != "---"
