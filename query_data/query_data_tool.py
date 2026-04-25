from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
MATCHES_CSV = DATA_DIR / "iplmatches.csv"
BALL_BY_BALL_CSV = DATA_DIR / "ipl2324.csv"
PLAYERS_CSV = DATA_DIR / "players-data-updated.csv"


matches_df = pd.read_csv(MATCHES_CSV, encoding="utf-8")
ball_df = pd.read_csv(BALL_BY_BALL_CSV, encoding="utf-8", low_memory=False)
players_df = pd.read_csv(PLAYERS_CSV, encoding="latin1")

if "id" in matches_df.columns and "match_id" not in matches_df.columns:
    matches_df.rename(columns={"id": "match_id"}, inplace=True)

for _df in (matches_df, ball_df, players_df):
    if "_source_row" not in _df.columns:
        _df["_source_row"] = _df.index.astype(int) + 2  # CSV line number; row 1 is the header.

ALL_TEAMS = pd.concat([matches_df["team1"], matches_df["team2"]]).dropna().unique().tolist()

TEAM_ALIASES = {
    "csk": ["chennai super kings"],
    "mi": ["mumbai indians"],
    "kkr": ["kolkata knight riders"],
    "rcb": ["royal challengers bangalore", "royal challengers bengaluru"],
    "srh": ["sunrisers hyderabad"],
    "dc": ["delhi capitals"],
    "rr": ["rajasthan royals"],
    "pbks": ["punjab kings"],
    "kxip": ["punjab kings"],
    "gt": ["gujarat titans"],
    "lsg": ["lucknow super giants"],
}


def _norm(value: Any) -> str:
    text = str(value).lower().strip()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase = _norm(phrase)
    if not phrase:
        return False
    return re.search(r"\b" + re.escape(phrase) + r"\b", _norm(text)) is not None


def _extract_season(question: str) -> Optional[int]:
    match = re.search(r"\b(20[0-9]{2})\b", question)
    return int(match.group(1)) if match else None


def _extract_top_n(question: str, default: int = 5) -> int:
    q = _norm(question)
    match = re.search(r"\btop\s*(\d+)\b", q)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(\d+)\s+(?:players|batters|batsmen|bowlers|scorers|wicket takers)\b", q)
    if match:
        return int(match.group(1))
    return default


def _error(message: str) -> Dict[str, Any]:
    return {
        "result": None,
        "columns": [],
        "row_count": 0,
        "source": None,
        "error": message,
    }


def _source_meta(
    dataset: str,
    source_df: Optional[pd.DataFrame],
    filters: Optional[Dict[str, Any]] = None,
    calculation: Optional[str] = None,
    max_rows: int = 25,
) -> Dict[str, Any]:
    if source_df is None:
        source_df = pd.DataFrame()

    row_count_used = int(len(source_df))
    rows_used: List[int] = []
    if "_source_row" in source_df.columns:
        rows_used = [int(row) for row in source_df["_source_row"].dropna().head(max_rows).tolist()]

    meta: Dict[str, Any] = {
        "dataset": dataset,
        "filters": filters or {},
        "row_count_used": row_count_used,
        "rows_used": rows_used,
        "citation": f"{dataset}; {row_count_used} matching data row(s)",
    }
    if calculation:
        meta["calculation"] = calculation
    if row_count_used > max_rows:
        meta["rows_used_note"] = f"showing first {max_rows} matching CSV data rows"
    return meta


def _to_result(
    value: Any,
    *,
    dataset: str,
    source_df: Optional[pd.DataFrame] = None,
    filters: Optional[Dict[str, Any]] = None,
    calculation: Optional[str] = None,
) -> Dict[str, Any]:
    if isinstance(value, pd.DataFrame):
        output_df = value.copy()
        if "_source_row" in output_df.columns:
            output_df = output_df.drop(columns=["_source_row"])
        output_df = output_df.reset_index(drop=True)
        return {
            "result": output_df.to_dict(orient="records"),
            "columns": output_df.columns.tolist(),
            "row_count": int(len(output_df)),
            "source": _source_meta(dataset, source_df if source_df is not None else value, filters, calculation),
        }

    if isinstance(value, pd.Series):
        output_df = value.reset_index()
        output_df.columns = [str(col) for col in output_df.columns]
        return {
            "result": output_df.to_dict(orient="records"),
            "columns": output_df.columns.tolist(),
            "row_count": int(len(output_df)),
            "source": _source_meta(dataset, source_df, filters, calculation),
        }

    return {
        "result": value,
        "columns": [],
        "row_count": 1,
        "source": _source_meta(dataset, source_df, filters, calculation),
    }


def _player_aliases(player_name: str, candidates_df: pd.DataFrame) -> List[str]:
    aliases = {_norm(player_name)}
    rows = candidates_df[candidates_df["player_name"].astype(str).str.lower() == str(player_name).lower()]

    for _, row in rows.iterrows():
        for col in ("player_name", "player_full_name", "player_name2"):
            if col in row and pd.notna(row[col]):
                alias = _norm(row[col])
                if alias:
                    aliases.add(alias)
                    parts = alias.split()
                    if parts:
                        aliases.add(parts[-1])

    parts = _norm(player_name).split()
    if parts:
        aliases.add(parts[-1])
    return sorted(aliases, key=len, reverse=True)


def _find_player_in_query(question: str, column: str) -> Optional[str]:
    q = _norm(question)
    names = sorted(ball_df[column].dropna().astype(str).unique().tolist())
    matches = []

    for name in names:
        for alias in _player_aliases(name, players_df):
            if len(alias) < 3:
                continue
            if re.search(r"\b" + re.escape(alias) + r"\b", q):
                matches.append((len(alias), name))
                break

    if not matches:
        return None

    matches.sort(reverse=True)
    return matches[0][1]


def _find_teams_in_query(question: str) -> List[str]:
    q = _norm(question)
    found: List[str] = []

    for abbr, full_names in TEAM_ALIASES.items():
        if re.search(r"\b" + re.escape(abbr) + r"\b", q):
            for full_name in full_names:
                for team in ALL_TEAMS:
                    if _norm(team) == full_name:
                        found.append(team)

    for team in ALL_TEAMS:
        team_norm = _norm(team)
        tokens = [token for token in team_norm.split() if len(token) >= 4]
        if team_norm in q or (tokens and all(token in q for token in tokens)):
            found.append(team)

    deduped: List[str] = []
    seen = set()
    for team in found:
        if team not in seen:
            seen.add(team)
            deduped.append(team)
    return deduped


def _team_participation_mask(df: pd.DataFrame, teams: List[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for team in teams:
        team_norm = _norm(team)
        team_mask = (
            df["team1"].astype(str).str.lower().str.contains(team_norm, regex=False, na=False)
            | df["team2"].astype(str).str.lower().str.contains(team_norm, regex=False, na=False)
        )
        mask &= team_mask
    return mask


def _winner_mask(df: pd.DataFrame, teams: List[str]) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for team in teams:
        mask |= df["winner"].astype(str).str.lower().str.contains(_norm(team), regex=False, na=False)
    return mask


def _is_final_query(question: str) -> bool:
    return bool(re.search(r"\bfinal\b", _norm(question)))


def _get_final_match(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("match_type", "stage", "type", "phase"):
        if col in df.columns:
            final_df = df[df[col].astype(str).str.lower().str.contains("final", na=False)].copy()
            if not final_df.empty:
                return final_df

    if "date" in df.columns:
        dated = df.copy()
        dated["_date_parsed"] = pd.to_datetime(dated["date"], dayfirst=True, errors="coerce")
        latest = dated["_date_parsed"].max()
        return dated[dated["_date_parsed"] == latest].drop(columns=["_date_parsed"])

    if "match_id" in df.columns:
        return df[df["match_id"] == df["match_id"].max()].copy()

    return df.tail(1).copy()


def _select_match_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        col
        for col in ("match_id", "season", "date", "team1", "team2", "winner", "venue", "match_type", "_source_row")
        if col in df.columns
    ]
    return df[columns].copy()


def query_data(question: str) -> Dict[str, Any]:
    """Answer IPL structured-data questions using safe pandas filters and aggregations.

    Use this tool for IPL match results, season summaries, team win counts, player runs,
    strike rates, wickets, top scorers, top wicket takers, and player profile fields.
    The input should be a short natural-language question. The output is always a dict
    with result, columns, row_count, and source metadata suitable for citations.
    """
    q = _norm(question)
    season = _extract_season(q)
    teams_found = _find_teams_in_query(q)

    if re.search(r"\b(how many|count|number of)\b", q) and re.search(r"\bmatch(?:es)?\b", q) and re.search(r"\b(win|won|wins)\b", q):
        df = matches_df.copy()
        filters: Dict[str, Any] = {}
        if season:
            df = df[df["season"] == season]
            filters["season"] = season
        if not teams_found:
            return _error("No team name found for match win count.")
        source_df = df[_winner_mask(df, teams_found)].copy()
        filters["winner"] = teams_found
        return _to_result(
            int(len(source_df)),
            dataset=MATCHES_CSV.name,
            source_df=source_df,
            filters=filters,
            calculation="count matching winner rows",
        )

    if re.search(r"\b(total\s+)?runs?\b", q) and not re.search(r"\b(top|most|highest|leading|list|team runs|runs by team)\b", q):
        player = _find_player_in_query(q, "batter")
        if player:
            df = ball_df.copy()
            filters = {"batter": player}
            if season:
                df = df[df["season"] == season]
                filters["season"] = season
            source_df = df[df["batter"].astype(str).str.lower() == player.lower()].copy()
            total = int(source_df["runs_batter"].sum())
            return _to_result(
                total,
                dataset=BALL_BY_BALL_CSV.name,
                source_df=source_df,
                filters=filters,
                calculation="sum runs_batter",
            )

    if re.search(r"\bstrike\s*rate\b", q):
        player = _find_player_in_query(q, "batter")
        if not player:
            return _error("Player name not found for strike-rate query.")
        df = ball_df.copy()
        filters = {"batter": player}
        if season:
            df = df[df["season"] == season]
            filters["season"] = season
        source_df = df[df["batter"].astype(str).str.lower() == player.lower()].copy()
        total_runs = int(source_df["runs_batter"].sum())
        balls_faced = int(source_df["valid_ball"].sum())
        if balls_faced == 0:
            return _error(f"No ball data found for {player}.")
        result = {
            "player": player,
            "runs": total_runs,
            "balls_faced": balls_faced,
            "strike_rate": round((total_runs / balls_faced) * 100, 2),
        }
        return _to_result(
            result,
            dataset=BALL_BY_BALL_CSV.name,
            source_df=source_df,
            filters=filters,
            calculation="runs_batter / valid_ball * 100",
        )

    if re.search(r"\bwickets?\b", q) and not re.search(r"\b(top|most|highest|leading|list)\b", q):
        bowler = _find_player_in_query(q, "bowler")
        if bowler:
            df = ball_df.copy()
            filters = {"bowler": bowler, "bowler_wicket": 1}
            if season:
                df = df[df["season"] == season]
                filters["season"] = season
            source_df = df[
                (df["bowler"].astype(str).str.lower() == bowler.lower())
                & (df["bowler_wicket"] == 1)
            ].copy()
            wickets = int(source_df["bowler_wicket"].sum())
            return _to_result(
                wickets,
                dataset=BALL_BY_BALL_CSV.name,
                source_df=source_df,
                filters=filters,
                calculation="sum bowler_wicket",
            )

    if re.search(r"\b(top|most|highest|leading)\b", q) and re.search(r"\brun(?:s|ners?| scorers?)?\b", q):
        n = _extract_top_n(q)
        df = ball_df.copy()
        filters: Dict[str, Any] = {"top_n": n}
        if season:
            df = df[df["season"] == season]
            filters["season"] = season
        result = (
            df.groupby("batter", as_index=False)["runs_batter"]
            .sum()
            .sort_values("runs_batter", ascending=False)
            .head(n)
            .rename(columns={"batter": "player", "runs_batter": "total_runs"})
        )
        return _to_result(
            result,
            dataset=BALL_BY_BALL_CSV.name,
            source_df=df,
            filters=filters,
            calculation="group by batter; sum runs_batter; sort descending",
        )

    if re.search(r"\b(top|most|highest|leading)\b", q) and re.search(r"\bwickets?\b", q):
        n = _extract_top_n(q)
        df = ball_df.copy()
        filters = {"top_n": n, "bowler_wicket": 1}
        if season:
            df = df[df["season"] == season]
            filters["season"] = season
        wicket_df = df[df["bowler_wicket"] == 1].copy()
        result = (
            wicket_df.groupby("bowler", as_index=False)["bowler_wicket"]
            .sum()
            .sort_values("bowler_wicket", ascending=False)
            .head(n)
            .rename(columns={"bowler_wicket": "wickets"})
        )
        return _to_result(
            result,
            dataset=BALL_BY_BALL_CSV.name,
            source_df=wicket_df,
            filters=filters,
            calculation="filter bowler_wicket=1; group by bowler; count wickets",
        )

    if re.search(r"\b(team runs|runs by team|total runs.*team)\b", q):
        df = ball_df.copy()
        filters: Dict[str, Any] = {}
        if season:
            df = df[df["season"] == season]
            filters["season"] = season
        result = (
            df.groupby("batting_team", as_index=False)["runs_batter"]
            .sum()
            .sort_values("runs_batter", ascending=False)
            .rename(columns={"batting_team": "team", "runs_batter": "total_runs"})
        )
        return _to_result(
            result,
            dataset=BALL_BY_BALL_CSV.name,
            source_df=df,
            filters=filters,
            calculation="group by batting_team; sum runs_batter",
        )

    is_h2h = bool(re.search(r"\b(between|vs|versus|against|head to head|head-to-head)\b", q))
    if is_h2h and len(teams_found) >= 2:
        df = matches_df.copy()
        filters = {"teams": teams_found[:2]}
        if season:
            df = df[df["season"] == season]
            filters["season"] = season
        source_df = df[_team_participation_mask(df, teams_found[:2])].copy()
        if source_df.empty:
            return _error(f"No matches found between {teams_found[0]} and {teams_found[1]}.")
        if re.search(r"\b(winner|won|win|wins?|record)\b", q):
            result = source_df["winner"].value_counts().reset_index()
            result.columns = ["team", "wins"]
            return _to_result(
                result,
                dataset=MATCHES_CSV.name,
                source_df=source_df,
                filters=filters,
                calculation="value counts of winner",
            )
        return _to_result(
            _select_match_columns(source_df),
            dataset=MATCHES_CSV.name,
            source_df=source_df,
            filters=filters,
        )

    if _is_final_query(q) and re.search(r"\b(winner|won|win|champion|result)\b", q):
        df = matches_df.copy()
        filters: Dict[str, Any] = {"match_type": "Final"}
        if season:
            df = df[df["season"] == season]
            filters["season"] = season
        source_df = _get_final_match(df)
        if source_df.empty:
            return _error("Final match not found.")
        return _to_result(
            _select_match_columns(source_df),
            dataset=MATCHES_CSV.name,
            source_df=source_df,
            filters=filters,
        )

    if re.search(r"\b(winner|won|win)\b", q):
        df = matches_df.copy()
        filters: Dict[str, Any] = {}
        if season:
            df = df[df["season"] == season]
            filters["season"] = season
        if teams_found:
            df = df[_team_participation_mask(df, teams_found)].copy()
            filters["teams"] = teams_found
        if _is_final_query(q):
            df = _get_final_match(df)
            filters["match_type"] = "Final"
        if df.empty:
            return _error("No matches found for the given criteria.")
        return _to_result(
            _select_match_columns(df),
            dataset=MATCHES_CSV.name,
            source_df=df,
            filters=filters,
        )

    if re.search(r"\b(season|matches? in|results? in|fixtures)\b", q):
        if not season:
            return _error("Season not found for season-results query.")
        source_df = matches_df[matches_df["season"] == season].copy()
        if source_df.empty:
            return _error(f"No matches found for season {season}.")
        return _to_result(
            _select_match_columns(source_df),
            dataset=MATCHES_CSV.name,
            source_df=source_df,
            filters={"season": season},
        )

    if re.search(r"\b(role|style|bat|bowl|field|position|info|details?)\b", q):
        name_matches = []
        for _, row in players_df.iterrows():
            aliases = []
            for col in ("player_name", "player_full_name", "player_name2"):
                if pd.notna(row.get(col)):
                    aliases.append(_norm(row[col]))
            if any(alias and re.search(r"\b" + re.escape(alias) + r"\b", q) for alias in aliases):
                name_matches.append(row.name)
        if not name_matches:
            return _error("Player name not found for player-info query.")
        source_df = players_df.loc[name_matches].copy()
        columns = ["player_name", "bat_style", "bowl_style", "field_pos", "player_full_name", "_source_row"]
        columns = [col for col in columns if col in source_df.columns]
        return _to_result(
            source_df[columns],
            dataset=PLAYERS_CSV.name,
            source_df=source_df,
            filters={"player_rows": len(source_df)},
        )

    return _error("Cannot answer from IPL structured dataset.")
