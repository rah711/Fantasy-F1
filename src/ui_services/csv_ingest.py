"""CSV ingestion for the weekly owner workflow.

Parsers are lenient about column names (the user assembles CSVs by hand
via Perplexity / Comet) but strict about types after parsing.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class IngestResult:
    ok: bool
    rows: int = 0
    saved_path: str | None = None
    updated_config: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    parsed: pd.DataFrame | None = None


def _parse_csv_text(text: str) -> pd.DataFrame | None:
    if not text or not text.strip():
        return None
    return pd.read_csv(StringIO(text.strip()))


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


# ---------------------------------------------------------------------------
# Price CSV ingest (drivers + constructors)
# ---------------------------------------------------------------------------

_DRIVER_CODE_COLS = ["code", "driver_code", "driver", "asset", "dr", "abbr"]
_CTOR_CODE_COLS = ["code", "constructor_id", "constructor", "team", "team_id", "asset", "cr"]
_PRICE_COLS = ["price", "value", "cost", "current_price"]


def ingest_driver_prices(cfg: dict[str, Any], csv_text: str) -> IngestResult:
    df = _parse_csv_text(csv_text)
    if df is None or df.empty:
        return IngestResult(ok=False, errors=["Driver prices CSV is empty"])

    code_col = _find_col(df, _DRIVER_CODE_COLS)
    price_col = _find_col(df, _PRICE_COLS)
    if not code_col:
        return IngestResult(ok=False, errors=[f"Could not find driver code column. Got: {list(df.columns)}"])
    if not price_col:
        return IngestResult(ok=False, errors=[f"Could not find price column. Got: {list(df.columns)}"])

    out = copy.deepcopy(cfg)
    target = out.setdefault("prices", {}).setdefault("drivers", {})
    if not isinstance(target, dict):
        return IngestResult(ok=False, errors=["config.prices.drivers is not a dict"])

    driver_lookup = _build_driver_name_lookup(cfg)
    warnings: list[str] = []
    updated = 0
    for i, row in df.iterrows():
        raw = str(row[code_col]).strip()
        if not raw or raw.lower() == "nan":
            continue
        code = _resolve_driver(raw, driver_lookup) or raw.upper()
        if code not in target:
            warnings.append(f"Unknown driver '{raw}' — not in config.prices.drivers (skipped)")
            continue
        try:
            price = float(row[price_col])
        except (TypeError, ValueError):
            warnings.append(f"Invalid price for {code}: {row[price_col]!r}")
            continue
        target[code]["price"] = round(price, 3)
        updated += 1

    return IngestResult(ok=True, rows=updated, updated_config=out, warnings=warnings, parsed=df)


def ingest_constructor_prices(cfg: dict[str, Any], csv_text: str) -> IngestResult:
    df = _parse_csv_text(csv_text)
    if df is None or df.empty:
        return IngestResult(ok=False, errors=["Constructor prices CSV is empty"])

    code_col = _find_col(df, _CTOR_CODE_COLS)
    price_col = _find_col(df, _PRICE_COLS)
    if not code_col:
        return IngestResult(ok=False, errors=[f"Could not find constructor column. Got: {list(df.columns)}"])
    if not price_col:
        return IngestResult(ok=False, errors=[f"Could not find price column. Got: {list(df.columns)}"])

    out = copy.deepcopy(cfg)
    target = out.setdefault("prices", {}).setdefault("constructors", {})
    if not isinstance(target, dict):
        return IngestResult(ok=False, errors=["config.prices.constructors is not a dict"])

    team_lookup = _build_team_name_lookup(cfg)
    warnings: list[str] = []
    updated = 0
    for i, row in df.iterrows():
        raw = str(row[code_col]).strip()
        if not raw or raw.lower() == "nan":
            continue
        cid = _resolve_team(raw, team_lookup)
        if cid is None or cid not in target:
            warnings.append(f"Unknown constructor '{raw}' — not in config.prices.constructors (skipped)")
            continue
        try:
            price = float(row[price_col])
        except (TypeError, ValueError):
            warnings.append(f"Invalid price for {cid}: {row[price_col]!r}")
            continue
        target[cid]["price"] = round(price, 3)
        updated += 1

    return IngestResult(ok=True, rows=updated, updated_config=out, warnings=warnings, parsed=df)


# ---------------------------------------------------------------------------
# Race + qualifying results ingest
# ---------------------------------------------------------------------------

_POS_COLS = ["position", "pos", "place", "finish", "finishing_position"]
_DRIVER_RES_COLS = ["driver_code", "driver", "code", "abbr", "name"]
_TEAM_RES_COLS = ["team", "constructor", "constructor_id", "team_id"]
_POINTS_COLS = ["points", "pts"]
_FL_COLS = ["fastest_lap", "fl", "fastest"]
_DOTD_COLS = ["dotd", "driver_of_the_day", "driver_of_day"]
_DNF_COLS = ["dnf", "retired", "out", "status", "time / retired", "time/retired", "time"]
_GAIN_COLS = ["positions_gained", "gained", "delta", "places_gained"]
_DNF_TOKENS = {"dnf", "retired", "ret", "dsq", "nc", "dq", "dns", "dnq"}


def _build_driver_name_lookup(cfg: dict[str, Any]) -> dict[str, str]:
    """Returns lowercased name fragment → driver code (e.g. {'antonelli': 'ANT', 'kimi antonelli': 'ANT'})."""
    out: dict[str, str] = {}
    for code, meta in (cfg.get("prices", {}).get("drivers", {}) or {}).items():
        full = str(meta.get("name", "")).strip()
        if not full:
            continue
        out[full.lower()] = code
        # Last name (handles "Antonelli", "Russell", "Hamilton")
        last = full.split()[-1].lower()
        out.setdefault(last, code)
        # Code itself, in case the CSV already uses codes
        out[code.lower()] = code
    return out


def _build_team_name_lookup(cfg: dict[str, Any]) -> dict[str, str]:
    """Returns lowercased name → constructor id (e.g. {'mercedes': 'mercedes', 'red bull': 'red_bull'})."""
    out: dict[str, str] = {}
    for cid, meta in (cfg.get("prices", {}).get("constructors", {}) or {}).items():
        out[cid.lower()] = cid
        out[cid.lower().replace("_", " ")] = cid  # 'red_bull' → 'red bull'
        full = str(meta.get("name", "")).strip()
        if full:
            out[full.lower()] = cid
            # Strip common suffix words ("Red Bull Racing" → "red bull")
            stripped = full.lower().replace(" racing", "").replace(" f1 team", "").strip()
            out.setdefault(stripped, cid)
    return out


def _resolve_driver(raw: str, lookup: dict[str, str]) -> str | None:
    if not raw:
        return None
    key = raw.strip().lower()
    if key in lookup:
        return lookup[key]
    # Try last word as last name
    last = key.split()[-1] if key.split() else key
    if last in lookup:
        return lookup[last]
    return None


def _resolve_team(raw: str, lookup: dict[str, str]) -> str | None:
    if not raw:
        return None
    key = raw.strip().lower()
    if key in lookup:
        return lookup[key]
    # Normalize spaces/hyphens/underscores and try again
    norm = key.replace("-", " ").replace("_", " ").strip()
    if norm in lookup:
        return lookup[norm]
    return None


def _normalize_results_df(
    df: pd.DataFrame, kind: str, cfg: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Standardize result columns to: position, driver_code, team_id, points, fastest_lap, dotd, dnf, gained.

    When `cfg` is provided, driver names and team display names are resolved
    against `prices.drivers` / `prices.constructors`, so a CSV with values like
    "Antonelli" / "Mercedes" is mapped to "ANT" / "mercedes".
    """
    warnings: list[str] = []
    pos = _find_col(df, _POS_COLS)
    drv = _find_col(df, _DRIVER_RES_COLS)
    if not pos or not drv:
        raise ValueError(
            f"{kind}: required columns missing. Need position + driver. Got: {list(df.columns)}"
        )

    driver_lookup = _build_driver_name_lookup(cfg) if cfg else {}
    team_lookup = _build_team_name_lookup(cfg) if cfg else {}

    out = pd.DataFrame()
    out["position"] = pd.to_numeric(df[pos], errors="coerce").astype("Int64")

    raw_drivers = df[drv].astype(str).str.strip()
    if driver_lookup:
        resolved = raw_drivers.apply(lambda x: _resolve_driver(x, driver_lookup))
        unresolved = raw_drivers[resolved.isna()].tolist()
        if unresolved:
            warnings.append(
                f"{kind}: could not resolve driver(s): {', '.join(sorted(set(unresolved)))} — left blank"
            )
        out["driver_code"] = resolved.fillna("").astype(str)
    else:
        out["driver_code"] = raw_drivers.str.upper()

    team_col = _find_col(df, _TEAM_RES_COLS)
    if team_col:
        raw_teams = df[team_col].astype(str).str.strip()
        if team_lookup:
            resolved_t = raw_teams.apply(lambda x: _resolve_team(x, team_lookup))
            unresolved_t = raw_teams[resolved_t.isna()].tolist()
            if unresolved_t:
                warnings.append(
                    f"{kind}: could not resolve team(s): {', '.join(sorted(set(unresolved_t)))} — left blank"
                )
            out["team_id"] = resolved_t.fillna("").astype(str)
        else:
            out["team_id"] = (
                raw_teams.str.lower().str.replace(" ", "_").str.replace("-", "_")
            )
    else:
        out["team_id"] = pd.NA
        warnings.append(f"{kind}: no team column found — team_id left blank")

    pts = _find_col(df, _POINTS_COLS)
    if pts:
        out["points"] = pd.to_numeric(df[pts], errors="coerce")
    else:
        out["points"] = pd.NA

    fl = _find_col(df, _FL_COLS)
    out["fastest_lap"] = df[fl].astype(bool) if fl else False

    dotd = _find_col(df, _DOTD_COLS)
    out["dotd"] = df[dotd].astype(bool) if dotd else False

    dnf = _find_col(df, _DNF_COLS)
    if dnf:
        s = df[dnf].astype(str).str.lower()
        out["dnf"] = s.apply(lambda v: any(tok in v for tok in _DNF_TOKENS))
    else:
        out["dnf"] = False

    gain = _find_col(df, _GAIN_COLS)
    out["positions_gained"] = pd.to_numeric(df[gain], errors="coerce") if gain else pd.NA

    out = out[out["driver_code"].astype(str).str.len() > 0]
    return out, warnings


def _save_results_csv(df: pd.DataFrame, project_root: Path, round_number: int, kind: str) -> Path:
    out_dir = project_root / "data" / "fantasy" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"round_{int(round_number):02d}_{kind}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def save_price_snapshot(
    csv_text: str,
    project_root: str | Path,
    round_number: int,
    kind: str,
) -> Path | None:
    """Archive the raw uploaded prices CSV to data/fantasy/prices/round_NN_{kind}.csv.

    `kind` is "drivers" or "constructors". Returns None for empty input.
    """
    if not csv_text or not csv_text.strip():
        return None
    out_dir = Path(project_root) / "data" / "fantasy" / "prices"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"round_{int(round_number):02d}_{kind}.csv"
    out_path.write_text(csv_text)
    return out_path


def ingest_race_results(
    csv_text: str,
    project_root: str | Path,
    round_number: int,
    cfg: dict[str, Any] | None = None,
) -> IngestResult:
    df = _parse_csv_text(csv_text)
    if df is None or df.empty:
        return IngestResult(ok=False, errors=["Race results CSV is empty"])
    try:
        cleaned, warnings = _normalize_results_df(df, "race", cfg=cfg)
    except ValueError as e:
        return IngestResult(ok=False, errors=[str(e)])
    saved = _save_results_csv(cleaned, Path(project_root), round_number, "race")
    return IngestResult(ok=True, rows=len(cleaned), saved_path=str(saved), warnings=warnings, parsed=cleaned)


def ingest_qualifying_results(
    csv_text: str,
    project_root: str | Path,
    round_number: int,
    cfg: dict[str, Any] | None = None,
) -> IngestResult:
    df = _parse_csv_text(csv_text)
    if df is None or df.empty:
        return IngestResult(ok=False, errors=["Qualifying results CSV is empty"])
    try:
        cleaned, warnings = _normalize_results_df(df, "qualifying", cfg=cfg)
    except ValueError as e:
        return IngestResult(ok=False, errors=[str(e)])
    saved = _save_results_csv(cleaned, Path(project_root), round_number, "qualifying")
    return IngestResult(ok=True, rows=len(cleaned), saved_path=str(saved), warnings=warnings, parsed=cleaned)


# ---------------------------------------------------------------------------
# Per-team-points helper (computes what the user's locked-in team scored)
# ---------------------------------------------------------------------------


def compute_team_points_for_round(
    project_root: str | Path,
    round_number: int,
    drivers: list[str],
    constructors: list[str],
    drs_boost: str | None,
) -> dict[str, Any]:
    """Compute fantasy-style points for the user's team for a given round.

    Uses the saved race results CSV. Returns a breakdown per driver + total.
    Returns {} if the race results file does not exist.
    """
    root = Path(project_root)
    race_path = root / "data" / "fantasy" / "results" / f"round_{int(round_number):02d}_race.csv"
    if not race_path.exists():
        return {}
    race = pd.read_csv(race_path)
    rows = []
    total = 0.0
    drs = (drs_boost or "").upper()
    for code in [str(d).upper() for d in drivers]:
        match = race[race["driver_code"] == code]
        if match.empty:
            rows.append({"driver": code, "points": None, "drs_doubled": code == drs})
            continue
        pts = float(match.iloc[0].get("points") or 0.0)
        if code == drs:
            pts *= 2.0
        total += pts
        rows.append({"driver": code, "points": pts, "drs_doubled": code == drs})
    return {
        "round": int(round_number),
        "drivers": rows,
        "constructors": [str(c).lower() for c in constructors],
        "total_driver_points": total,
        "race_results_path": str(race_path),
    }
