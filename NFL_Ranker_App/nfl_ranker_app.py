# app.py
import streamlit as st
import pandas as pd
import altair as alt
import sqlite3, re, datetime as dt, random
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Iterable


# --- GitHub sync helpers ---
import os, base64, json, requests

GH_OWNER  = st.secrets.get("GH_OWNER",  os.getenv("GH_OWNER"))
GH_REPO   = st.secrets.get("GH_REPO",   os.getenv("GH_REPO"))
GH_BRANCH = st.secrets.get("GH_BRANCH", os.getenv("GH_BRANCH", "main"))
GH_TOKEN  = st.secrets.get("GH_TOKEN",  os.getenv("GH_TOKEN"))

def _gh_enabled() -> bool:
    return all([GH_OWNER, GH_REPO, GH_BRANCH, GH_TOKEN])

def _gh_headers():
    return {"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github+json"}

def _gh_get_sha(path: str) -> Optional[str]:
    url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{path}?ref={GH_BRANCH}"
    r = requests.get(url, headers=_gh_headers())
    if r.status_code == 200:
        try:
            return r.json().get("sha")
        except Exception:
            return None
    return None

def gh_upsert_file(repo_path: str, bytes_content: bytes, message: str):
    """Create or update a file at repo_path on GH_BRANCH."""
    url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{repo_path}"
    payload = {
        "message": message,
        "branch": GH_BRANCH,
        "content": base64.b64encode(bytes_content).decode("ascii"),
    }
    sha = _gh_get_sha(repo_path)
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=_gh_headers(), data=json.dumps(payload), timeout=30)
    r.raise_for_status()

def gh_sync_to_repo(paths_to_commit: dict[str, Path], message: str):
    """paths_to_commit: {'data/games.csv': Path('/app/data/games.csv'), ...}"""
    if not _gh_enabled():
        return
    for repo_path, local_path in paths_to_commit.items():
        gh_upsert_file(repo_path, local_path.read_bytes(), message)
        
# -------------------- PATHS --------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CANDIDATES = [
    BASE_DIR / "data",           # app folder / data
    BASE_DIR.parent / "data",    # repo root / data
    Path.cwd() / "data",         # current working dir / data
]
for p in CANDIDATES:
    if (p / "games.csv").exists() and (p / "events.csv").exists() and (p / "teams.csv").exists():
        DATA_DIR = p
        break
else:
    # Don't create a new empty folder—fail loudly so you can fix placement
    import streamlit as st
    st.error("Couldn't find data/ with games.csv, events.csv, teams.csv next to your app or repo root.")
    st.stop()
DB_PATH = BASE_DIR / "storage" / "nfl.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

UPCOMING_CSV = DATA_DIR / "games_upcoming.csv"

# -------------------- DB CONNECTION (WAL + timeout) --------------------
DB_WRITE_LOCK = Lock()

def connect_db():
    con = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA busy_timeout=30000;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

# -------------------- CONSTANTS --------------------
IMPORTANCE_K = {"NCR":10, "CR":15, "DR":25, "WC":35, "DV":40, "CC":50, "SB":60}
LEAGUES = {"AP","N","AA","A"}
LEAGUE_FULL = {"N": "NFL", "AA": "AAFC", "A": "AFL", "AP": "APFA"}

def norm_league(lg: Optional[str]) -> Optional[str]:
    if lg is None: return None
    s = str(lg).strip().upper().replace(".", "")
    if s in ("N", "NFL"): return "N"
    if s in ("AA", "AAFC"): return "AA"
    if s in ("A", "AFL"): return "A"
    if s in ("AP", "APFA"): return "AP"
    return s or None

INDEPENDENT_ID = "ID000"
EVENT_ORDER = ["LEAVE","UNMERGE","MERGE","SWITCH","REJOIN","RENAME","JOIN"]
GAME_ID_RE = re.compile(r"^S([0-9A-F]{4})W([0-9A-F]{2})G([0-9A-F]{2})(AP|N|AA|A)$")

# -------------------- HELPERS --------------------
def parse_game_id(game_id: str):
    s = str(game_id).strip()
    m = GAME_ID_RE.match(s)
    if not m:
        return None
    season_hex, week_hex, seq_hex, lg = m.groups()
    return {"season": int(season_hex, 16), "week": int(week_hex, 16), "seq": int(seq_hex, 16), "league_indicator": lg}

def safe_int(x, default=None):
    try:
        return int(str(x).strip())
    except Exception:
        return default

def _division_key_row(r) -> tuple[str, str]:
    conf = (str(r.get("conference_display","")).strip()
            or str(r.get("conference_id","")).strip() or "")
    div  = (str(r.get("division_display","")).strip()
            or str(r.get("division_id","")).strip() or "")
    return conf, div

def _root_of_team_id(team_id: str, fr_map: dict[str,set[str]]) -> Optional[str]:
    team_id = str(team_id).strip()
    # Deterministically pick the first root whose members include this team ID
    for root, members in fr_map.items():
        if team_id in members:
            return root
    return None

def active_nfl_divisions_for_season(season: int,
                                    divisions_df: pd.DataFrame,
                                    events_df: pd.DataFrame,
                                    teams_df: pd.DataFrame,
                                    registry) -> tuple[list[dict], dict[str,str], dict[str,str]]:
    """
    Returns:
      divisions: list of {conf, div, roots: [franchise_root], team_ids: [teamID in this season]}
      root_to_teamid: franchise_root -> the active teamID for this season
      root_to_label: franchise_root -> team display name (for this season)
    Only uses NFL ('N') rows of divisions.csv for the season.
    """
    season = int(season)
    sub = divisions_df[
        (divisions_df["season"].apply(lambda x: safe_int(x)) == season) &
        (divisions_df["league_indicator"].map(norm_league) == "N")
    ].copy()
    if sub.empty:
        return [], {}, {}

    fr_map, _, _ = build_franchise_index(events_df, teams_df)
    disp_map = registry.get(season, {}).get("display", {})

    # group by division
    groups = {}
    for _, r in sub.iterrows():
        conf, div = _division_key_row(r)
        tid = str(r.get("team_id","")).strip()
        root = _root_of_team_id(tid, fr_map)
        if not tid or not root:
            continue
        key = (conf, div)
        g = groups.setdefault(key, {"conf": conf, "div": div, "roots": [], "team_ids": []})
        g["roots"].append(root)
        g["team_ids"].append(tid)

    # Prepare maps for quick labels
    root_to_teamid: dict[str, str] = {}
    root_to_label: dict[str, str] = {}
    for g in groups.values():
        for root, tid in zip(g["roots"], g["team_ids"]):
            root_to_teamid[root] = tid
            root_to_label[root] = disp_map.get(tid, tid)

    # Sort divisions by conference then division label; ensure each has 4 members
    divisions = sorted(groups.values(), key=lambda x: (x["conf"], x["div"]))
    return divisions, root_to_teamid, root_to_label

@st.cache_data(ttl=60)
def _load_all_nfl_games_for_h2h() -> pd.DataFrame:
    # lean frame for speed
    with connect_db() as con:
        g = pd.read_sql_query(
            "SELECT team_a_ID, team_a_score, team_b_ID, team_b_score "
            "FROM games WHERE league_indicator = 'N'",
            con
        )
    # normalize numeric
    if not g.empty:
        g["team_a_score"] = pd.to_numeric(g["team_a_score"], errors="coerce").fillna(0).astype(int)
        g["team_b_score"] = pd.to_numeric(g["team_b_score"], errors="coerce").fillna(0).astype(int)
        for c in ("team_a_ID", "team_b_ID"):
            g[c] = g[c].astype(str)
    return g

def h2h_map_for_selected_franchise(selected_root: str,
                                   opponent_roots: set[str],
                                   events_df: pd.DataFrame,
                                   teams_df: pd.DataFrame) -> dict[str, dict]:
    """
    For one selected franchise (root) compute all-time NFL head-to-head vs each
    opponent root in `opponent_roots`. Uses franchise membership across eras.
    Returns: opp_root -> {"W":int,"L":int,"T":int,"PD":int}
    """
    fr_map, _, _ = build_franchise_index(events_df, teams_df)
    mine = set(fr_map.get(selected_root, set()))
    if not mine:
        return {}

    # Build a mapping from every known team id to the opponent root it belongs to (for the active set)
    id_to_opp: dict[str, str] = {}
    for opp in opponent_roots:
        mem = fr_map.get(opp, set())
        for tid in mem:
            if tid not in mine:
                id_to_opp.setdefault(tid, opp)  # first wins, deterministic enough

    out: dict[str, dict] = {opp: {"W":0, "L":0, "T":0, "PD":0} for opp in opponent_roots}
    g = _load_all_nfl_games_for_h2h()
    if g.empty:
        return out

    for _, row in g.iterrows():
        a, b = row["team_a_ID"], row["team_b_ID"]
        sa, sb = int(row["team_a_score"]), int(row["team_b_score"])

        # Case 1: my franchise is A, opponent is B
        if a in mine and b in id_to_opp:
            opp = id_to_opp[b]
            if sa > sb: out[opp]["W"] += 1
            elif sa < sb: out[opp]["L"] += 1
            else: out[opp]["T"] += 1
            out[opp]["PD"] += (sa - sb)
            continue

        # Case 2: my franchise is B, opponent is A
        if b in mine and a in id_to_opp:
            opp = id_to_opp[a]
            if sb > sa: out[opp]["W"] += 1
            elif sb < sa: out[opp]["L"] += 1
            else: out[opp]["T"] += 1
            out[opp]["PD"] += (sb - sa)

    return out

def _holder_is_selected(rec: dict) -> Optional[bool]:
    """Return True if selected franchise holds the cup, False if opponent does, None if perfectly tied."""
    w, l, t, pd = rec.get("W",0), rec.get("L",0), rec.get("T",0), rec.get("PD",0)
    if w > l: return True
    if l > w: return False
    # wins tied -> PD tiebreak
    if pd > 0: return True
    if pd < 0: return False
    return None  # perfectly tied, no holder

def _format_record(rec: dict) -> str:
    return f"{rec.get('W',0)}–{rec.get('L',0)}–{rec.get('T',0)} • PD {rec.get('PD',0):+d}"

# -------------------- LOAD CSVs --------------------
# --- optional cups file (pairwise custom names) ---
CUPS_CSV = DATA_DIR / "cups.csv"

def load_cups_csv() -> pd.DataFrame:
    """
    Optional file to customize cup names between franchise pairs.
    Columns:
      franchise_a, franchise_b, cup_name
    The values for franchise_* are franchise roots (see Franchise Stats tab).
    If missing, we will auto-name as '<Team A> – <Team B> Cup'.
    """
    cols = ["franchise_a", "franchise_b", "cup_name"]
    if not CUPS_CSV.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(CUPS_CSV, dtype=str).fillna("")
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]

# --- optional, for pre-1933 or corrections ---
LEAGUE_CHAMPIONS_CSV = DATA_DIR / "league_champions.csv"

def load_league_champions_csv() -> pd.DataFrame:
    """
    Optional file to list league champions (e.g., pre-1933 when there's no 'SB' game).
    Expected columns (strings allowed):
      season, league_indicator, team_id, team_display, title_label, notes
    - season: int year
    - league_indicator: AP, N, AA, A (maps via norm_league)
    - team_id: champion team ID for that season (preferred)
    - team_display: OPTIONAL fallback display name to resolve the ID
    - title_label: OPTIONAL (if blank, we’ll show '<League> Champion')
    - notes: OPTIONAL
    """
    cols = ["season","league_indicator","team_id","team_display","title_label","notes"]
    if not LEAGUE_CHAMPIONS_CSV.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(LEAGUE_CHAMPIONS_CSV, dtype=str).fillna("")
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["league_indicator"] = df["league_indicator"].map(lambda x: (norm_league(x) or ""))
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]

# --- at the top near other paths ---
DIVISIONS_CSV = DATA_DIR / "divisions.csv"

def load_divisions_csv() -> pd.DataFrame:
    cols = ["season","league_indicator","conference_id","conference_display",
            "division_id","division_display","team_id",
            "is_division_champion","is_conference_champion","notes"]
    if not DIVISIONS_CSV.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(DIVISIONS_CSV, dtype=str).fillna("")
    # types
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["league_indicator"] = df["league_indicator"].map(lambda x: (norm_league(x) or ""))
    for c in ["is_division_champion","is_conference_champion"]:
        if c not in df.columns: df[c] = "0"
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).clip(0,1)
    # keep only expected cols
    for c in cols:
        if c not in df.columns: df[c] = ""
    return df[cols]

def load_csvs():
    games_path  = DATA_DIR / "games.csv"
    events_path = DATA_DIR / "events.csv"
    teams_path  = DATA_DIR / "teams.csv"
    # Load once at top-level near other CSV loads (after load_csvs()):

    missing_files = [p.name for p in [games_path, events_path, teams_path] if not p.exists()]
    if missing_files:
        st.error(f"Missing required CSV file(s) in ./data: {', '.join(missing_files)}")
        st.info("Expected: data/games.csv, data/events.csv, data/teams.csv")
        st.stop()

    games  = pd.read_csv(games_path, dtype=str).fillna("")
    events = pd.read_csv(events_path, dtype=str).fillna("")
    teams  = pd.read_csv(teams_path, dtype=str).fillna("")

    if "active To" in teams.columns and "activeTo" not in teams.columns:
        teams = teams.rename(columns={"active To": "activeTo"})

    required_games = {"league_indicator","season","week","game_ID","importance",
                      "team_a_display","team_a_ID","team_a_score",
                      "team_b_display","team_b_ID","team_b_score"}
    required_events = {"season","type","teamIDs"}
    required_teams  = {"teamID","displayName","activeFrom","activeTo"}

    for col in ("newID","league","fromLeague","toLeague"):
        if col not in events.columns:
            events[col] = ""

    missing_g = sorted(list(required_games - set(games.columns)))
    missing_e = sorted(list(required_events - set(events.columns)))
    missing_t = sorted(list(required_teams  - set(teams.columns)))
    if missing_g or missing_e or missing_t:
        if missing_g: st.error(f"[games.csv] Missing columns: {', '.join(missing_g)}"); st.write("Found columns:", list(games.columns))
        if missing_e: st.error(f"[events.csv] Missing columns: {', '.join(missing_e)}"); st.write("Found columns:", list(events.columns))
        if missing_t: st.error(f"[teams.csv] Missing columns: {', '.join(missing_t)}"); st.write("Found columns:", list(teams.columns))
        st.stop()

    # Normalize numeric-looking columns in games scores (do NOT coerce 'week' label)
    for c in ("team_a_score","team_b_score"):
        if c in games.columns:
            games[c] = games[c].apply(lambda x: safe_int(x, x))

    return games, events, teams

def franchise_division_titles(fr_root: str, divisions_df: pd.DataFrame,
                              events_df: pd.DataFrame, teams_df: pd.DataFrame):
    """
    Returns:
      {
        "total": int,  # unique seasons this franchise won a division
        "by_division": pd.DataFrame with columns ["Division","Titles"]
      }
    """
    empty = {"total": 0, "by_division": pd.DataFrame(columns=["Division","Titles"])}

    if divisions_df.empty:
        return empty

    fr_map, _, _ = build_franchise_index(events_df, teams_df)
    members = set(fr_map.get(fr_root, set()))
    if not members:
        return empty

    # Only rows with a division and one of the franchise's member teams
    df = divisions_df[
        (divisions_df["team_id"].isin(members)) &
        (divisions_df["division_id"].str.len() > 0)
    ].copy()

    wins = df[df["is_division_champion"] == 1]
    if wins.empty:
        return empty

    # Count titles by division label; sort by total desc then name asc
    agg = (wins.groupby(["division_id", "division_display"])["season"]
              .nunique()
              .reset_index(name="Titles")
              .sort_values(["Titles", "division_display"], ascending=[False, True]))

    by_div = agg.rename(columns={"division_display": "Division"}).drop(columns=["division_id"])
    total = int(wins["season"].nunique())

    return {"total": total, "by_division": by_div}

# -------- Upcoming CSV helpers (separate store; never affects ratings) --------
UPCOMING_COLS = ["upcoming_ID","league_indicator","season","week_order","week_label","importance",
                 "team_a_display","team_a_ID","team_b_display","team_b_ID"]

def load_upcoming_csv() -> pd.DataFrame:
    if not UPCOMING_CSV.exists():
        df = pd.DataFrame(columns=UPCOMING_COLS)
        df.to_csv(UPCOMING_CSV, index=False)
        return df
    df = pd.read_csv(UPCOMING_CSV, dtype=str).fillna("")
    # ensure all columns exist
    for c in UPCOMING_COLS:
        if c not in df.columns:
            df[c] = ""
    # normalize numeric-ish columns
    if not df.empty:
        df["upcoming_ID"] = df["upcoming_ID"].apply(lambda x: safe_int(x, x))
        df["season"] = df["season"].apply(lambda x: safe_int(x, x))
        df["week_order"] = df["week_order"].apply(lambda x: safe_int(x, x))
    return df[UPCOMING_COLS]

def save_upcoming_csv(df: pd.DataFrame):
    df = df.copy()
    if "upcoming_ID" in df.columns:
        try:
            df["upcoming_ID"] = pd.to_numeric(df["upcoming_ID"], errors="coerce").astype("Int64")
        except Exception:
            pass
    df.to_csv(UPCOMING_CSV, index=False)

def next_upcoming_id(df: pd.DataFrame) -> int:
    if df.empty or "upcoming_ID" not in df.columns or df["upcoming_ID"].isna().all():
        return 1
    try:
        return int(pd.to_numeric(df["upcoming_ID"], errors="coerce").max()) + 1
    except Exception:
        return 1

# -------------------- REGISTRY (active teams, display names) --------------------
def build_season_registry(events_df: pd.DataFrame, teams_df: pd.DataFrame):
    def is_active(team_row, season):
        af = safe_int(team_row["activeFrom"], 9999)
        at_raw = str(team_row["activeTo"]).strip().lower()
        at = 9999 if at_raw == "present" else safe_int(at_raw, 9999)
        return af <= season <= at

    seasons = set()
    if not events_df.empty:
        seasons |= set(events_df["season"].apply(lambda x: safe_int(x)).dropna().astype(int).unique())
    if not teams_df.empty:
        min_af = teams_df["activeFrom"].apply(lambda x: safe_int(x)).dropna().astype(int).min()
        seasons |= set(range(min_af, dt.date.today().year + 1))
    if not seasons:
        seasons = {dt.date.today().year}

    base_display = dict(zip(teams_df["teamID"], teams_df["displayName"]))
    registry = {}

    for season in sorted(seasons):
        active = {r["teamID"] for _, r in teams_df.iterrows() if is_active(r, season)}
        canonical = {t: t for t in active}
        display   = {t: base_display.get(t, t) for t in active}

        e_season = events_df[events_df["season"].apply(lambda x: safe_int(x)) == season].copy()
        if not e_season.empty:
            for ev_type in EVENT_ORDER:
                for _, ev in e_season[e_season["type"] == ev_type].iterrows():
                    ids = [x.strip() for x in str(ev.get("teamIDs","")).split(",") if x.strip()]
                    new_id = str(ev.get("newID","")).strip()

                    if ev_type == "JOIN":
                        for t in ids: active.add(t); canonical[t] = t; display[t] = base_display.get(t, t)
                    elif ev_type == "LEAVE":
                        for t in ids: active.discard(t); canonical.pop(t, None); display.pop(t, None)
                    elif ev_type == "MERGE":
                        if new_id:
                            for t in ids: active.discard(t); canonical[t] = new_id
                            active.add(new_id); canonical[new_id] = new_id
                            display[new_id] = base_display.get(new_id, display.get(new_id, new_id))
                    elif ev_type == "UNMERGE":
                        outs = [x.strip() for x in str(new_id).split(",") if x.strip()]
                        for o in outs: active.add(o); canonical[o] = o; display[o] = base_display.get(o, display.get(o, o))
                        for p in ids:
                            if p not in outs: active.discard(p); canonical.pop(p, None); display.pop(p, None)
                    elif ev_type == "REJOIN":
                        for t in ids: active.add(t); canonical[t] = t; display[t] = base_display.get(t, t)
                    elif ev_type == "RENAME":
                        if ids and new_id:
                            old = ids[0]
                            if old in active: active.discard(old)
                            active.add(new_id)
                            for k, v in list(canonical.items()):
                                if v == old: canonical[k] = new_id
                            canonical[new_id] = new_id
                            display[new_id] = base_display.get(new_id, display.get(old, new_id))
                            display.pop(old, None)
                    elif ev_type == "SWITCH":
                        pass

        registry[season] = {"active": active, "canonical": canonical, "display": display}
    return registry

def canonicalize(registry, season, team_id):
    season = int(season)
    if season in registry and team_id in registry[season]["canonical"]:
        return registry[season]["canonical"][team_id]
    return team_id

def prune_to_active_season(ratings: Dict[str, float], season: int, registry) -> Dict[str, float]:
    act = set(registry.get(int(season), {}).get("active", set()))
    if not act:
        return ratings
    pruned: Dict[str, float] = {}
    for t, r in list(ratings.items()):
        t_can = canonicalize(registry, int(season), str(t))
        if t_can in act:
            # If multiple old IDs map to same canonical, keep the max rating (arbitrary tie-break)
            pruned[t_can] = max(pruned.get(t_can, r), r)
    return pruned

# -------------------- SQLITE --------------------
def init_db():
    with connect_db() as con:
        cur = con.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_ID TEXT PRIMARY KEY,
            league_indicator TEXT,
            season INTEGER,
            week_order INTEGER,
            week_label TEXT,
            importance TEXT,
            team_a_display TEXT, team_a_ID TEXT, team_a_score INTEGER,
            team_b_display TEXT, team_b_ID TEXT, team_b_score INTEGER
        );""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            team_ID TEXT,
            season INTEGER,
            week INTEGER,
            rating REAL,
            PRIMARY KEY (team_ID, season, week)
        );""")
        con.commit()

def mirror_into_db(games_df: pd.DataFrame):
    with DB_WRITE_LOCK:
        with connect_db() as con:
            cur = con.cursor()
            cur.execute("DELETE FROM games;")
            for _, g in games_df.iterrows():
                parsed = parse_game_id(g["game_ID"])
                if not parsed:
                    raise ValueError(f"Invalid game_ID format: {g['game_ID']}")
                season_i = parsed["season"]
                week_ord = parsed["week"]
                league_i = parsed["league_indicator"]
                week_label = str(g["week"])
                sa = safe_int(g["team_a_score"], 0)
                sb = safe_int(g["team_b_score"], 0)
                cur.execute("""INSERT OR REPLACE INTO games
                    (game_ID, league_indicator, season, week_order, week_label, importance,
                     team_a_display, team_a_ID, team_a_score, team_b_display, team_b_ID, team_b_score)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (g["game_ID"], league_i, season_i, week_ord, week_label, g["importance"],
                     g["team_a_display"], g["team_a_ID"], sa, g["team_b_display"], g["team_b_ID"], sb)
                )
            con.commit()

def read_games_db():
    with connect_db() as con:
        df = pd.read_sql_query("SELECT * FROM games ORDER BY season, week_order, game_ID", con)
    return df

def latest_week(season):
    with connect_db() as con:
        cur = con.cursor()
        cur.execute("SELECT COALESCE(MAX(week_order),0) FROM games WHERE season = ?", (int(season),))
        w = cur.fetchone()[0] or 0
    return int(w)

def ratings_table(season, week):
    with connect_db() as con:
        df = pd.read_sql_query(
            "SELECT * FROM ratings WHERE season = ? AND week = ? ORDER BY rating DESC",
            con, params=(int(season), int(week))
        )
    return df

def _invalidate_cached_dropdowns():
    try:
        st.cache_data.clear()
    except Exception:
        pass

def pretty_ratings_df(season:int, week:int, registry) -> pd.DataFrame:
    df = ratings_table(season, week)
    if df.empty:
        return df
    disp_map = registry.get(int(season), {}).get("display", {})
    df["Team"] = df["team_ID"].map(lambda t: disp_map.get(t, t))
    df = df[["Team", "rating"]].rename(columns={"rating": "Rating"})
    df.index = pd.RangeIndex(start=1, stop=len(df) + 1, step=1)
    return df

from collections import Counter, defaultdict

def leagues_participated_for_franchise(
    fr_root: str,
    events_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    registry,
    divisions_df: pd.DataFrame
) -> list[str]:
    """
    Returns the list of leagues a franchise actually belonged to:
      1) Prefer divisions.csv league_indicator for (season, team)
      2) Else use events-derived league map for that season
      3) Else fall back to that team's own games in that season (excluding SB) and take the mode
    """
    fr_map, _, _ = build_franchise_index(events_df, teams_df)
    members = set(fr_map.get(fr_root, set()))
    if not members:
        return []

    # Build quick (season, canonical_team_id) -> league code from divisions.csv
    div_lookup: dict[tuple[int, str], str] = {}
    if isinstance(divisions_df, pd.DataFrame) and not divisions_df.empty:
        sub = divisions_df.copy()
        sub["season"] = pd.to_numeric(sub["season"], errors="coerce").astype("Int64")
        for _, r in sub.iterrows():
            s = r["season"]
            if pd.isna(s): 
                continue
            s = int(s)
            tid_raw = str(r.get("team_id", "")).strip()
            if not tid_raw:
                continue
            lg = norm_league(r.get("league_indicator", ""))
            if not lg:
                continue
            # canonicalize for the season
            tid = canonicalize(registry, s, tid_raw)
            div_lookup[(s, tid)] = lg

    # Pull games once for fallback mode
    with connect_db() as con:
        g = pd.read_sql_query(
            "SELECT season, importance, league_indicator, team_a_ID, team_b_ID "
            "FROM games ORDER BY season",
            con
        )
    if not g.empty:
        g["season"] = pd.to_numeric(g["season"], errors="coerce").fillna(0).astype(int)

    # Build seasons to check from registry keys (covers all active years)
    seasons_all = sorted(int(s) for s in registry.keys())
    leagues_codes: set[str] = set()

    for s in seasons_all:
        # Map from events/games context: team -> league (best effort)
        league_of = build_league_map_for_season_preseason(s, registry, events_df)

        # Active members this season
        active_ids = registry.get(s, {}).get("active", set())
        active_members = (members & set(active_ids))
        if not active_members:
            continue

        for tid_raw in active_members:
            tid = canonicalize(registry, s, tid_raw)
            if tid == INDEPENDENT_ID:
                continue

            lg = None
            # 1) divisions.csv (strongest signal)
            lg = div_lookup.get((s, tid))
            # 2) events-derived map
            if not lg:
                lg = norm_league(league_of.get(tid)) if league_of else None
            # 3) fallback to this team's own games in that season (excluding SB)
            if not lg and not g.empty:
                rows = g[(g["season"] == s) &
                         ((g["team_a_ID"] == tid_raw) | (g["team_b_ID"] == tid_raw))].copy()
                if not rows.empty:
                    rows = rows[rows["importance"].str.upper() != "SB"]  # avoid cross-league SB label
                    if not rows.empty:
                        votes = [norm_league(x) for x in rows["league_indicator"].tolist() if norm_league(x)]
                        if votes:
                            lg = Counter(votes).most_common(1)[0][0]

            if lg:
                leagues_codes.add(lg)

    # Map to full names for display
    return sorted(LEAGUE_FULL.get(c, c) for c in leagues_codes)

# -------------------- RECORDS & SIM HELPERS --------------------
def current_record_to_week(season:int, up_to_week:int, registry) -> dict[str, dict]:
    """
    Compute each team's current record from actual games up to and including up_to_week.
    Returns: records[team_ID] = {"W":int,"L":int,"T":int}
    """
    df = read_games_db()
    if df.empty:
        return {}
    df = df[(df["season"].astype(int) == int(season)) &
            (df["week_order"].astype(int) <= int(up_to_week))].copy()
    rec: dict[str, dict] = {}
    def bump(tid, w=0,l=0,t=0):
        d = rec.setdefault(tid, {"W":0,"L":0,"T":0})
        d["W"] += w; d["L"] += l; d["T"] += t
    for _, g in df.iterrows():
        a = canonicalize(registry, int(season), str(g["team_a_ID"]))
        b = canonicalize(registry, int(season), str(g["team_b_ID"]))
        sa = safe_int(g["team_a_score"], 0)
        sb = safe_int(g["team_b_score"], 0)
        if sa > sb:
            bump(a, w=1); bump(b, l=1)
        elif sa < sb:
            bump(a, l=1); bump(b, w=1)
        else:
            bump(a, t=1); bump(b, t=1)
    return rec

def simulate_full_season(
    q_season:int,
    start_week:int,
    registry,
    snapshot_ratings: pd.DataFrame,
    schedule_df: pd.DataFrame,
    mode: str = "Monte Carlo",
    runs: int = 500,
    tie_prob: float = 0.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run remaining-season sims starting AFTER start_week using upcoming schedule entries.
    - snapshot_ratings: df from ratings_table(season, week) at the chosen starting week
    - schedule_df: upcoming schedule rows for this season with week_order > start_week
    Returns a DataFrame with projected remaining wins and final ratings.
    """
    if schedule_df.empty or snapshot_ratings.empty:
        return pd.DataFrame()

    if seed is not None:
        random.seed(int(seed))

    base_ratings = {r["team_ID"]: float(r["rating"]) for _, r in snapshot_ratings.iterrows()}
    teams = set(base_ratings.keys())

    for _, r in schedule_df.iterrows():
        teams.add(canonicalize(registry, q_season, str(r["team_a_ID"])))
        teams.add(canonicalize(registry, q_season, str(r["team_b_ID"])))

    sched = []
    for _, r in schedule_df.iterrows():
        w = safe_int(r["week_order"], None)
        if w is None:
            continue
        a = canonicalize(registry, q_season, str(r["team_a_ID"]))
        b = canonicalize(registry, q_season, str(r["team_b_ID"]))
        imp = IMPORTANCE_K.get(str(r.get("importance","")).strip() or "CR", 15)
        sched.append((w, a, b, imp))
    sched.sort(key=lambda x: (x[0], x[1], x[2]))

    def p_win(ra, rb):
        return 1.0 / (1.0 + 10 ** (-(ra - rb) / 600.0))

    if mode == "Favorites win":
        ratings = dict(base_ratings)
        wins = {t: 0 for t in teams}
        ties = {t: 0 for t in teams}
        for w, a, b, imp in sched:
            ra, rb = ratings.get(a, 1000.0), ratings.get(b, 1000.0)
            pa = p_win(ra, rb)
            if tie_prob > 0 and random.random() < tie_prob:
                ra, rb = elo_update(ra, rb, 1, 1, imp)
                ratings[a], ratings[b] = ra, rb
                ties[a] += 1; ties[b] += 1
                continue
            if pa >= 0.5:
                ra, rb = elo_update(ra, rb, 1, 0, imp)
                wins[a] += 1
            else:
                ra, rb = elo_update(ra, rb, 0, 1, imp)
                wins[b] += 1
            ratings[a], ratings[b] = ra, rb

        out = []
        for t in sorted(teams):
            out.append({
                "team_ID": t,
                "exp_remaining_wins": float(wins.get(t, 0)),
                "exp_remaining_ties": float(ties.get(t, 0)),
                "final_rating_mean": ratings.get(t, 1000.0),
                "final_rating_sd": 0.0
            })
        return pd.DataFrame(out)

    runs = max(1, int(runs))
    agg_win = {t: 0.0 for t in teams}
    agg_tie = {t: 0.0 for t in teams}
    agg_rating_sum = {t: 0.0 for t in teams}
    agg_rating_sq = {t: 0.0 for t in teams}

    for _ in range(runs):
        ratings = dict(base_ratings)
        wins = {t: 0 for t in teams}
        ties = {t: 0 for t in teams}
        for w, a, b, imp in sched:
            ra, rb = ratings.get(a, 1000.0), ratings.get(b, 1000.0)
            if tie_prob > 0 and random.random() < tie_prob:
                ra, rb = elo_update(ra, rb, 1, 1, imp)
                ratings[a], ratings[b] = ra, rb
                ties[a] += 1; ties[b] += 1
                continue
            pa = p_win(ra, rb)
            if random.random() < pa:
                ra, rb = elo_update(ra, rb, 1, 0, imp)
                wins[a] += 1
            else:
                ra, rb = elo_update(ra, rb, 0, 1, imp)
                wins[b] += 1
            ratings[a], ratings[b] = ra, rb

        for t in teams:
            r = ratings.get(t, 1000.0)
            agg_win[t] += wins.get(t, 0)
            agg_tie[t] += ties.get(t, 0)
            agg_rating_sum[t] += r
            agg_rating_sq[t] += r * r

    rows = []
    for t in sorted(teams):
        mean_r = agg_rating_sum[t] / runs
        var_r = max(0.0, (agg_rating_sq[t] / runs) - (mean_r * mean_r))
        sd_r = var_r ** 0.5
        rows.append({
            "team_ID": t,
            "exp_remaining_wins": agg_win[t] / runs,
            "exp_remaining_ties": agg_tie[t] / runs,
            "final_rating_mean": mean_r,
            "final_rating_sd": sd_r
        })
    return pd.DataFrame(rows)

# --- long form helpers for charts ---
def ratings_long_df(season:int, registry) -> pd.DataFrame:
    with connect_db() as con:
        df = pd.read_sql_query(
            "SELECT season, week, team_ID, rating FROM ratings WHERE season = ?",
            con, params=(int(season),)
        )
    if df.empty:
        return df
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    disp_map = registry.get(int(season), {}).get("display", {})
    df["Team"] = df["team_ID"].map(lambda t: disp_map.get(t, t))
    wkmap = season_week_label_map(int(season))
    df["WeekLabel"] = df["week"].map(lambda w: wkmap.get(int(w), f"Week {int(w)}"))
    df["XLabel"] = df.apply(lambda r: f"{r['season']} {r['WeekLabel']}", axis=1)
    df["t"] = df["week"].astype(int)
    return df.sort_values(["week", "Team"])

def ratings_long_df_range(season_min:int, season_max:int, registry) -> pd.DataFrame:
    season_min, season_max = int(season_min), int(season_max)
    with connect_db() as con:
        df = pd.read_sql_query(
            "SELECT season, week, team_ID, rating FROM ratings WHERE season BETWEEN ? AND ?",
            con, params=(season_min, season_max)
        )
    if df.empty:
        return df
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    def season_disp(s, tid):
        disp_map = registry.get(int(s), {}).get("display", {})
        return disp_map.get(tid, tid)
    df["Team"] = [season_disp(s, t) for s, t in zip(df["season"], df["team_ID"])]
    last_names = (df.sort_values(["season", "week"]).groupby("team_ID")["Team"].last().to_dict())
    df["Legend"] = df["team_ID"].map(lambda tid: f"{last_names.get(tid, tid)} ({tid})")
    df["t"] = df["season"] * 100 + df["week"]
    m = multi_season_week_label_map(season_min, season_max)
    df["WeekLabel"] = [m.get((int(s), int(w)), f"Week {int(w)}") for s, w in zip(df["season"], df["week"])]
    df["XLabel"] = [f"{int(s)} {wl}" for s, wl in zip(df["season"], df["WeekLabel"])]
    return df.sort_values(["season", "week", "team_ID"])

def _format_week_label(week: int, raw_label: str | None) -> str:
    if int(week) == 0:
        return "Week 0 (events)"
    s = (raw_label or "").strip()
    if s.isdigit():
        return f"Week {int(s)}"
    return s if s else f"Week {int(week)}"

def season_week_label_map(season: int) -> dict[int, str]:
    with connect_db() as con:
        df = pd.read_sql_query(
            "SELECT week_order AS week, MIN(week_label) AS label "
            "FROM games WHERE season = ? GROUP BY week_order",
            con, params=(int(season),)
        )
    m = {int(r["week"]): _format_week_label(int(r["week"]), r["label"]) for _, r in df.iterrows()}
    m[0] = _format_week_label(0, None)
    return m

def multi_season_week_label_map(season_min: int, season_max: int) -> dict[tuple[int,int], str]:
    with connect_db() as con:
        df = pd.read_sql_query(
            "SELECT season, week_order AS week, MIN(week_label) AS label "
            "FROM games WHERE season BETWEEN ? AND ? GROUP BY season, week_order",
            con, params=(int(season_min), int(season_max))
        )
    m = {(int(r["season"]), int(r["week"])): _format_week_label(int(r["week"]), r["label"]) for _, r in df.iterrows()}
    for s in range(int(season_min), int(season_max)+1):
        m.setdefault((s, 0), _format_week_label(0, None))
    return m

# --- chart builder ---
def line_points_chart(plot_df, x_enc, color_field: str, color_title: str, y_scale, order_field: str, tooltips: list,
                      height: int = 360, show_points: bool = True):
    base = alt.Chart(plot_df).encode(
        x=x_enc,
        y=alt.Y("rating:Q", title="Rating", scale=y_scale),
        color=alt.Color(f"{color_field}:N", title=color_title),
        order=alt.Order(f"{order_field}:Q"),
        tooltip=tooltips,
    )
    line = base.mark_line(clip=True)
    if show_points:
        pts = base.mark_point(size=28, filled=True, opacity=0.85)
        return alt.layer(line, pts).properties(height=height).interactive()
    return line.properties(height=height).interactive()

@st.cache_data(ttl=30)
def build_timeline_index(s_min: int, s_max: int) -> pd.DataFrame:
    with connect_db() as con:
        tdf = pd.read_sql_query("""
            SELECT DISTINCT season, week
            FROM ratings
            WHERE season BETWEEN ? AND ?
            ORDER BY season, week
        """, con, params=(int(s_min), int(s_max)))
    tdf["t_cont"] = range(len(tdf))
    return tdf

def season_boundaries_df(timeline_df: pd.DataFrame) -> pd.DataFrame:
    if timeline_df.empty:
        return timeline_df
    return timeline_df[timeline_df["week"] == 0][["t_cont", "season"]].copy()

# --- Franchise index ---
@st.cache_data(ttl=30)
def build_franchise_index(events_df: pd.DataFrame, teams_df: pd.DataFrame):
    parent: dict[str, str] = {}
    def find(x: str) -> str:
        x = str(x).strip()
        if x not in parent: parent[x] = x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: str, b: str):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra

    ids: set[str] = set(teams_df["teamID"].astype(str))
    if not events_df.empty:
        for _, ev in events_df.iterrows():
            for t in [x.strip() for x in str(ev.get("teamIDs","")).split(",") if x.strip()]: ids.add(t)
            for n in [x.strip() for x in str(ev.get("newID","")).split(",") if x.strip()]: ids.add(n)

    if not events_df.empty:
        for _, ev in events_df[events_df["type"] == "RENAME"].iterrows():
            olds = [x.strip() for x in str(ev.get("teamIDs","")).split(",") if x.strip()]
            new  = str(ev.get("newID","")).strip()
            if olds and new: union(olds[0], new)

    class_members: dict[str, set[str]] = {}
    for t in ids: class_members.setdefault(find(t), set()).add(t)
    franchise_map: dict[str, set[str]] = {root: set(members) for root, members in class_members.items()}

    if not events_df.empty:
        for _, ev in events_df[events_df["type"] == "MERGE"].iterrows():
            srcs = [x.strip() for x in str(ev.get("teamIDs","")).split(",") if x.strip()]
            new  = str(ev.get("newID","")).strip()
            if not srcs or not new: continue
            new_root  = find(new)
            new_class = class_members.get(new_root, {new})
            for s in srcs:
                s_root = find(s)
                franchise_map.setdefault(s_root, set()).update(new_class)

    id_to_franchises: dict[str, set[str]] = {t: set() for t in ids}
    for fr_root, members in franchise_map.items():
        for t in members:
            id_to_franchises.setdefault(t, set()).add(fr_root)

    teams_idx = {row["teamID"]: row for _, row in teams_df.iterrows()}
    def to_year(v):
        s = str(v).strip().lower()
        if s == "present": return 9999
        try: return int(s)
        except: return -1
    def pick_label(root: str) -> str:
        members = list(franchise_map.get(root, []))
        best_id, best_to = None, -1
        for t in members:
            row = teams_idx.get(t)
            if row is None: continue
            y = to_year(row.get("activeTo", ""))
            if y > best_to:
                best_to, best_id = y, t
            if y == 9999:
                best_id = t; break
        if best_id is None: return f"{root} (franchise)"
        row_best = teams_idx.get(best_id)
        name = str(row_best.get("displayName", best_id)) if row_best is not None else str(best_id)
        return f"{name} (franchise)"
    franchise_labels: dict[str, str] = {root: pick_label(root) for root in franchise_map.keys()}
    return franchise_map, id_to_franchises, franchise_labels

@st.cache_data(ttl=30)
def get_seasons():
    with connect_db() as con:
        cur = con.cursor()
        cur.execute("""
            SELECT season FROM games
            UNION
            SELECT season FROM ratings
            ORDER BY season
        """)
        rows = cur.fetchall()
    return [int(r[0]) for r in rows]

@st.cache_data(ttl=30)
def get_weeks_for_season(season:int):
    with connect_db() as con:
        weeks_df = pd.read_sql_query(
            "SELECT DISTINCT week FROM ratings WHERE season = ? ORDER BY week", con, params=(int(season),)
        )
        if weeks_df.empty:
            return []
        records = []
        for w in weeks_df["week"].astype(int).tolist():
            if w == 0:
                label = "Week 0 (events)"
            else:
                lbl_df = pd.read_sql_query(
                    "SELECT MIN(week_label) AS week_label FROM games WHERE season = ? AND week_order = ?",
                    con, params=(int(season), int(w))
                )
                label = str(lbl_df.iloc[0]["week_label"]) if not lbl_df.empty and lbl_df.iloc[0]["week_label"] else str(w)
            records.append({"week_order": int(w), "week_label": label})
    return records

def ratings_is_empty() -> bool:
    with connect_db() as con:
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM ratings;")
        n = cur.fetchone()[0] or 0
    return n == 0

def get_end_of_prev_season_ratings(season:int) -> Dict[str, float]:
    with connect_db() as con:
        prev = int(season) - 1
        dfw = pd.read_sql_query("SELECT MAX(week) AS mw FROM ratings WHERE season = ?", con, params=(prev,))
        if dfw.empty or pd.isna(dfw.iloc[0]["mw"]):
            return {}
        mw = int(dfw.iloc[0]["mw"])
        df = pd.read_sql_query(
            "SELECT team_ID, rating FROM ratings WHERE season = ? AND week = ?",
            con, params=(prev, mw)
        )
        return {r["team_ID"]: float(r["rating"]) for _, r in df.iterrows()}

# -------------------- ELO --------------------
def elo_update(ra, rb, score_a, score_b, importance):
    if score_a > score_b:
        Wa, Wb = 1.0, 0.0
    elif score_a < score_b:
        Wa, Wb = 0.0, 1.0
    else:
        Wa, Wb = 0.5, 0.5
    dr = ra - rb
    We_a = 1.0 / (1.0 + 10 ** (-dr / 600.0))
    We_b = 1.0 - We_a
    ra_new = ra + importance * (Wa - We_a)
    rb_new = rb + importance * (Wb - We_b)
    return ra_new, rb_new

# -------------------- League mapping + Week-0 --------------------
def overlay_event_leagues_for_season(league_of: dict, season: int, events_df: pd.DataFrame) -> dict:
    if events_df.empty:
        return league_of
    evs = events_df[events_df["season"].apply(lambda x: safe_int(x)) == int(season)].copy()
    evs_non_switch = evs[evs["type"].isin(["LEAVE","UNMERGE","MERGE","REJOIN","RENAME","JOIN"])]
    for _, ev in evs_non_switch.iterrows():
        lg = norm_league(ev.get("league", ""))
        if not lg: continue
        ids = [x.strip() for x in str(ev.get("teamIDs","")).split(",") if x.strip()]
        for t in ids: league_of[t] = lg
        new_id = str(ev.get("newID","")).strip()
        if new_id:
            for c in [x.strip() for x in new_id.split(",") if x.strip()]:
                league_of[c] = lg
    evs_switch = evs[evs["type"] == "SWITCH"]
    for _, ev in evs_switch.iterrows():
        src = norm_league(ev.get("fromLeague", ""))
        if not src: continue
        ids = [x.strip() for x in str(ev.get("teamIDs","")).split(",") if x.strip()]
        for t in ids: league_of[t] = src
    return league_of

def build_league_map_for_season_preseason(season: int, registry, events_df: pd.DataFrame) -> Dict[str, str]:
    def _from_games(season_q: int) -> Dict[str, str]:
        with connect_db() as con:
            df = pd.read_sql_query(
                "SELECT league_indicator, team_a_ID, team_b_ID FROM games WHERE season = ?",
                con, params=(int(season_q),)
            )
        m: Dict[str, str] = {}
        if df.empty: return m
        for _, r in df.iterrows():
            lg = norm_league(r["league_indicator"])
            for t in (r["team_a_ID"], r["team_b_ID"]):
                t_can = canonicalize(registry, season_q, str(t))
                if t_can and t_can != INDEPENDENT_ID and lg:
                    m[t_can] = lg
        return m
    prev_map = _from_games(int(season) - 1) if season > 1900 else {}
    return overlay_event_leagues_for_season(prev_map, season, events_df)

def normalize_mean_to_1000(ratings: Dict[str, float], fixed: Optional[Iterable[str]] = None,
                           league_of: Optional[Dict[str, str]] = None, limit_leagues: Optional[Iterable[str]] = None):
    if not ratings: return ratings
    fixed_ids = set(fixed or [])
    if not league_of:
        N = len(ratings)
        if N == 0: return ratings
        sum_all = sum(ratings.values())
        adjustable = [k for k in ratings.keys() if k not in fixed_ids]
        M = len(adjustable)
        if M == 0: return ratings
        delta = (1000.0 * N - sum_all) / M
        for k in adjustable: ratings[k] = ratings[k] + delta
        return ratings
    leagues: Dict[str, list] = {}
    for team in ratings.keys():
        lg = league_of.get(team, "_UNK_")
        leagues.setdefault(lg, []).append(team)
    limit = set(limit_leagues) if limit_leagues is not None else None
    for lg, members in leagues.items():
        if limit is not None and lg not in limit: continue
        N = len(members)
        if N == 0: continue
        sum_l = sum(ratings[t] for t in members if t in ratings)
        adjustable = [t for t in members if t not in fixed_ids]
        M = len(adjustable)
        if M == 0: continue
        delta = (1000.0 * N - sum_l) / M
        for t in adjustable: ratings[t] = ratings[t] + delta
    return ratings

def apply_week0_events_to_ratings(ratings: Dict[str, float], season: int, events_df: pd.DataFrame,
                                  last_left: Dict[str, float], last_left_league: Dict[str, str], registry):
    if events_df.empty:
        return ratings
    evs = events_df[events_df["season"].apply(lambda x: safe_int(x)) == int(season)].copy()
    if evs.empty:
        return ratings
    league_of = build_league_map_for_season_preseason(int(season), registry, events_df)
    snapshots = {}

    # LEAVE
    from collections import Counter
    did_have_leave_rows = False
    leave_leagues = set()
    for _, ev in evs[evs["type"] == "LEAVE"].iterrows():
        ids = [x.strip() for x in str(ev.get("teamIDs", "")).split(",") if x.strip()]
        ev_lg = norm_league(ev.get("league", "")) or None
        did_have_leave_rows = True
        if not ev_lg:
            inferred = [norm_league(league_of.get(t)) for t in ids if league_of.get(t)]
            if inferred:
                ev_lg = Counter(inferred).most_common(1)[0][0]
        if ev_lg: leave_leagues.add(ev_lg)
        for t in ids:
            src_lg = ev_lg or norm_league(league_of.get(t))
            if t in ratings:
                last_left[t] = ratings[t]; last_left_league[t] = src_lg; ratings.pop(t, None)
            league_of.pop(t, None)
    if did_have_leave_rows and leave_leagues:
        normalize_mean_to_1000(ratings, fixed=None, league_of=league_of, limit_leagues=leave_leagues)
    snapshots["LEAVE"] = ratings.copy()

    # UNMERGE
    any_unmerge = False
    unmerge_children, unmerge_leagues = [], set()
    for _, ev in evs[evs["type"] == "UNMERGE"].iterrows():
        parents  = [x.strip() for x in str(ev.get("teamIDs", "")).split(",") if x.strip()]
        children = [x.strip() for x in str(ev.get("newID", "")).split(",") if x.strip()]
        if not parents or not children: continue
        for p in parents:
            base = ratings.get(p, 1000.0)
            parent_lg = norm_league(ev.get("league", "")) or norm_league(league_of.get(p))
            if parent_lg: unmerge_leagues.add(parent_lg)
            for c in children:
                ratings[c] = base; unmerge_children.append(c)
                if parent_lg: league_of[c] = parent_lg
            if p not in children:
                ratings.pop(p, None); league_of.pop(p, None)
            any_unmerge = True
    if any_unmerge and unmerge_leagues:
        normalize_mean_to_1000(ratings, fixed=set(unmerge_children), league_of=league_of, limit_leagues=unmerge_leagues)
    snapshots["UNMERGE"] = ratings.copy()

    # MERGE
    merge_source = snapshots.get("UNMERGE") or snapshots.get("LEAVE") or ratings
    any_merge, merged_new_ids, merge_leagues = False, [], set()
    for _, ev in evs[evs["type"] == "MERGE"].iterrows():
        ids = [x.strip() for x in str(ev.get("teamIDs", "")).split(",") if x.strip()]
        new_id = str(ev.get("newID", "")).strip()
        if not new_id or not ids: continue
        vals, missing, parent_leagues = [], [], []
        for t in ids:
            if t in merge_source: vals.append(merge_source[t])
            elif t in ratings:   vals.append(ratings[t])
            elif t in last_left: vals.append(last_left[t])
            else: missing.append(t)
            if t in league_of: parent_leagues.append(norm_league(league_of[t]))
        if missing: vals.extend([1000.0] * len(missing))
        avg = sum(vals) / len(vals) if vals else 1000.0
        for t in ids: ratings.pop(t, None); league_of.pop(t, None)
        ratings[new_id] = avg
        ev_lg = norm_league(ev.get("league", ""))
        if ev_lg:
            league_of[new_id] = ev_lg; merge_leagues.add(ev_lg)
        elif parent_leagues:
            league_of[new_id] = parent_leagues[0]; merge_leagues.update(parent_leagues)
        merged_new_ids.append(new_id); any_merge = True
    if any_merge and merge_leagues:
        normalize_mean_to_1000(ratings, fixed=set(merged_new_ids), league_of=league_of, limit_leagues=merge_leagues)
    snapshots["MERGE"] = ratings.copy()

    # SWITCH
    any_switch, switch_ids, moves, src_leagues = False, set(), [], set()
    for _, ev in evs[evs["type"] == "SWITCH"].iterrows():
        ids = [x.strip() for x in str(ev.get("teamIDs", "")).split(",") if x.strip()]
        src_hint = norm_league(ev.get("fromLeague", "")) or None
        dst_hint = norm_league(ev.get("toLeague", "") or ev.get("league", "")) or None
        for t in ids:
            curr_map = norm_league(league_of.get(t))
            src_lg = src_hint or curr_map; dst_lg = dst_hint
            moves.append((t, src_lg, dst_lg))
            if src_lg and curr_map == src_lg:
                league_of[t] = "_SWITCH_HOLD_"; switch_ids.add(t); src_leagues.add(src_lg); any_switch = True
    if any_switch and src_leagues:
        normalize_mean_to_1000(ratings, fixed=switch_ids, league_of=league_of, limit_leagues=src_leagues)
    for (t, src_lg, dst_lg) in moves:
        league_of[t] = (dst_lg or src_lg)
    snapshots["SWITCH"] = ratings.copy()

    # REJOIN
    any_rejoin, rejoin_ids, rejoin_leagues = False, [], set()
    for _, ev in evs[evs["type"] == "REJOIN"].iterrows():
        ids = [x.strip() for x in str(ev.get("teamIDs", "")).split(",") if x.strip()]
        ev_lg = norm_league(ev.get("league", "")) or None
        for t in ids:
            ratings[t] = last_left.get(t, 1000.0)
            lg = norm_league(ev_lg or last_left_league.get(t))
            if lg: league_of[t] = lg; rejoin_leagues.add(lg)
            rejoin_ids.append(t); any_rejoin = True
    if any_rejoin and rejoin_leagues:
        normalize_mean_to_1000(ratings, fixed=set(rejoin_ids), league_of=league_of, limit_leagues=rejoin_leagues)
    snapshots["REJOIN"] = ratings.copy()

    # RENAME
    for _, ev in evs[evs["type"] == "RENAME"].iterrows():
        olds   = [x.strip() for x in str(ev.get("teamIDs", "")).split(",") if x.strip()]
        new_id = str(ev.get("newID", "")).strip()
        if not olds or not new_id: continue
        for old in olds:
            if old in ratings: ratings[new_id] = ratings[old]; ratings.pop(old, None)
            if old in league_of: league_of[new_id] = norm_league(league_of[old]); league_of.pop(old, None)

    # JOIN
    for _, ev in evs[evs["type"] == "JOIN"].iterrows():
        ids = [x.strip() for x in str(ev.get("teamIDs", "")).split(",") if x.strip()]
        for t in ids: ratings[t] = ratings.get(t, 1000.0)
    return ratings

def baseline_ratings(season:int, week:int, registry, events_csv: pd.DataFrame) -> Dict[str, float]:
    """
    Use stored snapshot if it exists; otherwise synthesize preseason from
    end-of-previous-season ratings + Week-0 events.
    """
    snap = ratings_table(season, week)
    if not snap.empty:
        return {r["team_ID"]: float(r["rating"]) for _, r in snap.iterrows()}
    base = get_end_of_prev_season_ratings(season)
    if not base:
        return {}
    last_left, last_left_league = {}, {}
    snap = apply_week0_events_to_ratings(base.copy(), season, events_csv, last_left, last_left_league, registry)
    return prune_to_active_season(snap, season, registry)

# -------------------- RECOMPUTE --------------------
def recompute_ratings_from(season_start, week_start, registry):
    with DB_WRITE_LOCK:
        with connect_db() as con:
            cur = con.cursor()
            cur.execute("""
                DELETE FROM ratings WHERE (season > ?) OR (season = ? AND week >= ?);
            """, (season_start, season_start, week_start))

            prior = pd.read_sql_query("""
                SELECT * FROM ratings WHERE (season < ?) OR (season = ? AND week < ?)
            """, con, params=(season_start, season_start, week_start))
            ratings: Dict[str, float] = {r["team_ID"]: float(r["rating"]) for _, r in prior.iterrows()}

            events = pd.read_csv(DATA_DIR / "events.csv", dtype=str).fillna("")
            games = pd.read_sql_query("SELECT * FROM games ORDER BY season, week_order, game_ID", con)

            last_left: Dict[str, float] = {}
            last_left_league: Dict[str, str] = {}
            applied_week0_for_season, snapshotted_week0 = set(), set()
            BASE = 1000.0

            to_play = games[(games["season"] > season_start) |
                            ((games["season"] == season_start) & (games["week_order"] >= week_start))].copy()

            for (s, w), week_df in to_play.groupby(["season","week_order"], sort=True):
                s = int(s); w = int(w)
                if s not in applied_week0_for_season:
                    ratings = apply_week0_events_to_ratings(ratings, s, events, last_left, last_left_league, registry)
                    ratings = prune_to_active_season(ratings, s, registry)
                    applied_week0_for_season.add(s)

                if s not in snapshotted_week0:
                    # NEW: remove stale Week 0 rows for this season so defunct teams can’t linger
                    cur.execute("DELETE FROM ratings WHERE season = ? AND week = 0", (s,))
                    for t, r in ratings.items():
                        cur.execute(
                            "INSERT OR REPLACE INTO ratings (team_ID, season, week, rating) VALUES (?,?,?,?)",
                            (t, s, 0, float(r))
        )
                    snapshotted_week0.add(s)

                for _, g in week_df.iterrows():
                    a_raw, b_raw = g["team_a_ID"], g["team_b_ID"]
                    a = canonicalize(registry, s, a_raw)
                    b = canonicalize(registry, s, b_raw)
                    if a == INDEPENDENT_ID or b == INDEPENDENT_ID: continue
                    ra = ratings.get(a, BASE); rb = ratings.get(b, BASE)
                    sa = safe_int(g["team_a_score"], 0); sb = safe_int(g["team_b_score"], 0)
                    imp = IMPORTANCE_K[g["importance"]]
                    ra_new, rb_new = elo_update(ra, rb, sa, sb, imp)
                    ratings[a], ratings[b] = ra_new, rb_new

                for t, r in ratings.items():
                    cur.execute(
                        "INSERT OR REPLACE INTO ratings (team_ID, season, week, rating) VALUES (?,?,?,?)",
                        (t, s, w, float(r))
                    )
            con.commit()

# -------------------- FRANCHISE STATS --------------------
def _longest_true_streak(bools_in_order: Iterable[bool]) -> int:
    longest = cur = 0
    for v in bools_in_order:
        if v:
            cur += 1
            if cur > longest: longest = cur
        else:
            cur = 0
    return longest

def _playoff_category(season: int, importance: str, league_indicator: str | None = None) -> Optional[str]:
    """
    Map CSV 'importance' codes to historical playoff rounds.

    Rules requested:
      • 1966–1969: CC counts as League Championship (NFL or AFL Championship).
      • 1966–1969: DV in the NFL (not AFL) counts as Conference Championship.
      • 1950–1965: CC counts as Conference Championship.
      • Modern era: DV = Divisional Round (1966+), WC = Wildcard.
      • Pre-1950 CC falls back to Divisional Round (one-tier predecessor).

    Returns one of: "League Championship/Super Bowl", "Conference Championship",
                    "Divisional Round", "Wildcard Round", or None.
    """
    imp = (importance or "").strip().upper()
    lg = norm_league(league_indicator) if league_indicator else None
    y = int(season)

    # Always: SB is the league title line
    if imp == "SB":
        return "League Championship/Super Bowl"

    # Window: 1966–1969 (pre-merger Super Bowl era)
    if 1966 <= y <= 1969:
        if imp == "CC":
            # NFL or AFL Championship Game
            return "League Championship/Super Bowl"
        if imp == "DV" and lg == "N":  # NFL only
            # NFL Eastern/Western Conference Championships
            return "Conference Championship"
        # For AFL, DV (if it appears) is not a conference title — fall through.

    # 1950–1965: conferences exist; CC is the conf title
    if imp == "CC" and y >= 1950:
        return "Conference Championship"

    # Divisional Round exists as a true round from 1966 onward
    if imp == "DV":
        return "Divisional Round" if y >= 1966 else None

    if imp == "WC":
        return "Wildcard Round"

    # Before 1950, if CC shows up, treat it like the tier below the league title
    if imp == "CC" and y < 1950:
        return "Divisional Round"

    return None

def compute_franchise_stats(fr_root: str,
                            events_df: pd.DataFrame,
                            teams_df: pd.DataFrame,
                            registry,
                            divisions_df: pd.DataFrame):
    """
    Returns a dict with:
      - names_history: list of {Name, From, To}
      - leagues_participated: list[str] (full names)
      - best_rank, best_rank_rating, best_when (season, week, label, team_name)
      - worst_rank, worst_rank_rating, worst_when (...)
      - weeks_at_1, longest_1_streak, weeks_top5, longest_top5_streak
      - playoff_wins_df (DataFrame with Season, Round, Opponent, Score, Category, League)
      - playoff_counts (dict by Category)
      (plus back-compat aliases 'name_history', 'titles_df')
    """
    fr_map, _, fr_labels = build_franchise_index(events_df, teams_df)
    members = set(fr_map.get(fr_root, set()))

    # --- Name history (group by displayName with From/To) ---
    sub = teams_df[teams_df["teamID"].isin(members)].copy()
    def _to_year(v):
        s = str(v).strip().lower()
        if s == "present": return 9999
        try: return int(s)
        except: return -1
    name_rows = []
    if not sub.empty:
        grouped = sub.groupby("displayName", dropna=False)
        for name, g in grouped:
            y_from = min([_to_year(x) for x in g["activeFrom"].tolist()] or [None])
            y_to   = max([_to_year(x) for x in g["activeTo"].tolist()] or [None])
            name_rows.append({"Name": str(name), "From": y_from, "To": y_to})
    names_history = sorted(name_rows, key=lambda r: (r["From"] if r["From"] is not None else 9999, r["Name"]))

    # --- Leagues participated (from actual games) ---
    with connect_db() as con:
        gdf = pd.read_sql_query(
            "SELECT league_indicator, team_a_ID, team_b_ID FROM games",
            con
        )
    leagues_codes = set()
    for _, r in gdf.iterrows():
        if r["team_a_ID"] in members or r["team_b_ID"] in members:
            lg = norm_league(r["league_indicator"])
            if lg:
                leagues_codes.add(lg)
    leagues_participated = leagues_participated_for_franchise(fr_root, events_df, teams_df, registry, divisions_df)

    # --- Rankings across time (exclude season 1920 and week 0) ---
    with connect_db() as con:
        rdf = pd.read_sql_query(
            "SELECT season, week, team_ID, rating FROM ratings WHERE week > 0 AND season <> 1920",
            con
        )
    if rdf.empty:
        # Build empty shape to avoid KeyErrors later
        frweek_df = pd.DataFrame(columns=["season","week","team_ID","rating","pos","WeekLabel","Team"])
        best_rank = worst_rank = weeks_at_1 = longest_1 = weeks_top5 = longest_top5 = 0
        best_ctx = worst_ctx = (None,None,None,None)
        best_rt = worst_rt = None
    else:
        # Rank per (season, week)
        rdf["season"] = rdf["season"].astype(int)
        rdf["week"] = rdf["week"].astype(int)
        rdf["rating"] = rdf["rating"].astype(float)
        rdf["pos"] = rdf.groupby(["season","week"])["rating"].rank(ascending=False, method="min").astype(int)

        # Best rank among franchise members for each (season, week)
        mem = rdf[rdf["team_ID"].isin(members)].copy()
        if mem.empty:
            frweek_df = pd.DataFrame(columns=["season","week","team_ID","rating","pos","WeekLabel","Team"])
            best_rank = worst_rank = weeks_at_1 = longest_1 = weeks_top5 = longest_top5 = 0
            best_ctx = worst_ctx = (None,None,None,None)
            best_rt = worst_rt = None
        else:
            # pick the row achieving the min pos per (season, week)
            idx = mem.groupby(["season","week"])["pos"].idxmin()
            frweek_df = mem.loc[idx, ["season","week","team_ID","rating","pos"]].copy()

            # decorate with labels/names
            def _wklabel(s, w):
                return season_week_label_map(int(s)).get(int(w), f"Week {int(w)}")
            frweek_df["WeekLabel"] = [ _wklabel(s, w) for s, w in zip(frweek_df["season"], frweek_df["week"]) ]
            def _disp(s, tid):
                return registry.get(int(s), {}).get("display", {}).get(tid, tid)
            frweek_df["Team"] = [ _disp(s, t) for s, t in zip(frweek_df["season"], frweek_df["team_ID"]) ]
            frweek_df = frweek_df.sort_values(["season","week"]).reset_index(drop=True)

            # ---------- Best/Worst picking with tie-break on rating ----------
            # BEST: lowest rank; among those rows choose the row with MAX rating
            best_rank = int(frweek_df["pos"].min())
            best_pool = frweek_df[frweek_df["pos"] == best_rank]
            best_row = best_pool.loc[best_pool["rating"].idxmax()]

            # WORST: highest rank; among those rows choose the row with MIN rating
            worst_rank = int(frweek_df["pos"].max())
            worst_pool = frweek_df[frweek_df["pos"] == worst_rank]
            worst_row = worst_pool.loc[worst_pool["rating"].idxmin()]

            best_rt = float(best_row["rating"])
            worst_rt = float(worst_row["rating"])
            best_ctx = (int(best_row["season"]), int(best_row["week"]), str(best_row["WeekLabel"]), str(best_row["Team"]))
            worst_ctx = (int(worst_row["season"]), int(worst_row["week"]), str(worst_row["WeekLabel"]), str(worst_row["Team"]))

            # Streak metrics
            mask_1 = (frweek_df["pos"] == 1).tolist()
            mask_5 = (frweek_df["pos"] <= 5).tolist()
            weeks_at_1 = int(sum(mask_1))
            weeks_top5 = int(sum(mask_5))
            longest_1 = _longest_true_streak(mask_1)
            longest_top5 = _longest_true_streak(mask_5)

    # --- Playoff wins table (with CSV week label in 'Round') ---
    with connect_db() as con:
        g2 = pd.read_sql_query(
            "SELECT season, week_order, week_label, importance, league_indicator, "
            "team_a_display, team_a_ID, team_a_score, team_b_display, team_b_ID, team_b_score "
            "FROM games ORDER BY season, week_order",
            con
        )
    rows = []
    if not g2.empty:
        for _, r in g2.iterrows():
            season = safe_int(r["season"], None)
            if season is None:
                continue
            importance = str(r["importance"]).strip().upper()
            league_ind = norm_league(r["league_indicator"])
            cat = _playoff_category(int(season), importance)
            if not cat:
                continue  # not one of the four playoff rounds we track

            a_id, b_id = str(r["team_a_ID"]), str(r["team_b_ID"])
            a_win = safe_int(r["team_a_score"], 0) > safe_int(r["team_b_score"], 0)
            b_win = safe_int(r["team_b_score"], 0) > safe_int(r["team_a_score"], 0)

            winner_is_franchise = None
            opp = None
            score_txt = None

            if a_id in members and a_win:
                winner_is_franchise = True
                opp = str(r["team_b_display"])
                score_txt = f"{safe_int(r['team_a_score'],0)}–{safe_int(r['team_b_score'],0)}"
            elif b_id in members and b_win:
                winner_is_franchise = True
                opp = str(r["team_a_display"])
                score_txt = f"{safe_int(r['team_b_score'],0)}–{safe_int(r['team_a_score'],0)}"

            if winner_is_franchise:
                rows.append({
                    "Season": int(season),
                    "Round": str(r["week_label"]) if str(r["week_label"]).strip() else f"Week {int(r['week_order'])}",
                    "Category": cat,
                    "Opponent": opp,
                    "Score": score_txt,
                    "League": LEAGUE_FULL.get(norm_league(r["league_indicator"]), str(r["league_indicator"])),
                    "week_order": safe_int(r["week_order"], 0)
                })

    playoff_wins_df = pd.DataFrame(rows)
    if not playoff_wins_df.empty:
        playoff_wins_df = playoff_wins_df.sort_values(["Season","week_order","Category"]).drop(columns=["week_order"])
    playoff_counts = dict(playoff_wins_df["Category"].value_counts()) if not playoff_wins_df.empty else {}

    out = {
        "names_history": names_history,
        "leagues_participated": leagues_participated,
        "best_rank": best_rank,
        "best_rank_rating": best_rt,
        "best_when": best_ctx,     # (season, week, label, team_name)
        "worst_rank": worst_rank,
        "worst_rank_rating": worst_rt,
        "worst_when": worst_ctx,   # (season, week, label, team_name)
        "weeks_at_1": weeks_at_1,
        "longest_1_streak": longest_1,
        "weeks_top5": weeks_top5,
        "longest_top5_streak": longest_top5,
        "playoff_wins_df": playoff_wins_df,
        "playoff_counts": playoff_counts,
        "franchise_label": fr_labels.get(fr_root, fr_root),
    }

    # ---- back-compat aliases so existing UI keeps working ----
    out["name_history"] = out.get("names_history", [])
    out["titles_df"] = out.get("playoff_wins_df")
    out["leagues"] = out.get("leagues_participated", [])

    return out

def _to_roman(n: int) -> str:
    vals = [
        (1000,"M"), (900,"CM"), (500,"D"), (400,"CD"),
        (100,"C"), (90,"XC"), (50,"L"), (40,"XL"),
        (10,"X"), (9,"IX"), (5,"V"), (4,"IV"), (1,"I")
    ]
    out = []
    x = int(n)
    for v, sym in vals:
        while x >= v:
            out.append(sym)
            x -= v
    return "".join(out)

def _super_bowl_title_for_season(season: int) -> str:
    """Season 1966 -> SB I, …; special-case SB 50 to use '50' not 'L'."""
    num = int(season) - 1965
    if num == 50:
        return "Super Bowl 50"
    return f"Super Bowl {_to_roman(num)} Champion"

def franchise_titles_table(fr_root: str,
                           divisions_df: pd.DataFrame,
                           events_df: pd.DataFrame,
                           teams_df: pd.DataFrame,
                           registry, league_champs_df=None) -> pd.DataFrame:
    """
    One row per title the franchise has won:
      columns: Season, Level (League|Conference|Division), Title

    - Division titles come from divisions.csv (no league prefix, no 'Champion' suffix).
    - Conference titles come from divisions.csv (labeled '<Conference> Champion').
      If CSV already marks a conference champion, we DO NOT add another from games.
    - League/Super Bowl titles come from games with era rules via _playoff_category.
      SB uses roman numerals, except SB 50 uses 'Super Bowl 50'. Pre-SB era uses
      '<NFL/AAFC/AFL> Champion'.
    - Robust de-dup (normalize 'Champion' wording in Conf/Div).
    """
    import re

    rows = []

    # --- franchise membership ---
    fr_map, _, _ = build_franchise_index(events_df, teams_df)
    members = set(fr_map.get(fr_root, set()))
    if not members:
        return pd.DataFrame(columns=["Season", "Level", "Title"])

    def _safe_int(x, d=None):
        try:
            return int(str(x).strip())
        except Exception:
            return d

    # ---------- From divisions.csv ----------
    conf_marks: set[tuple[str, int]] = set()           # (canon_team_id, season)
    conf_label_map: dict[tuple[str, int], str] = {}    # (canon_team_id, season) -> conference_display

    if isinstance(divisions_df, pd.DataFrame) and not divisions_df.empty:
        sub = divisions_df[divisions_df["team_id"].isin(members)].copy()

        # Division champions — show just the division label (no 'Champion' suffix)
        dmask = (sub["is_division_champion"] == 1)
        for _, r in sub[dmask].iterrows():
            s = _safe_int(r.get("season"))
            if s is None: 
                continue
            div_label = (str(r.get("division_display","")).strip()
                         or str(r.get("division_id","")).strip())
            if div_label:
                rows.append({"Season": s, "Level": "Division", "Title": f"{div_label} Champion"})

        # Cache conference labels and record CSV-marked champs
        for _, r in sub.iterrows():
            s = _safe_int(r.get("season"))
            if s is None:
                continue
            tid = canonicalize(registry, s, str(r.get("team_id","")).strip())
            clabel = (str(r.get("conference_display","")).strip()
                      or str(r.get("conference_id","")).strip())
            if clabel:
                conf_label_map[(tid, s)] = clabel
            if int(_safe_int(r.get("is_conference_champion", 0), 0)) == 1:
                conf_marks.add((tid, s))

        # Add conference champs from CSV (single, consistent label)
        cmask = (sub["is_conference_champion"] == 1)
        for _, r in sub[cmask].iterrows():
            s = _safe_int(r.get("season"))
            if s is None: 
                continue
            clabel = (str(r.get("conference_display","")).strip()
                      or str(r.get("conference_id","")).strip())
            if clabel:
                rows.append({"Season": s, "Level": "Conference", "Title": f"{clabel} Champion"})

    # ---------- From games ----------
    with connect_db() as con:
        gdf = pd.read_sql_query(
            "SELECT season, week_order, week_label, importance, league_indicator, "
            "team_a_ID, team_a_score, team_b_ID, team_b_score "
            "FROM games ORDER BY season, week_order", con
        )

    if not gdf.empty:
        for _, g in gdf.iterrows():
            season = _safe_int(g["season"])
            if season is None:
                continue
            imp = str(g["importance"]).strip().upper()
            lg  = norm_league(g.get("league_indicator"))
            cat = _playoff_category(int(season), imp, lg)
            if cat not in ("League Championship/Super Bowl", "Conference Championship"):
                continue

            # Winner
            a_id, b_id = str(g["team_a_ID"]), str(g["team_b_ID"])
            sa, sb = _safe_int(g["team_a_score"], 0), _safe_int(g["team_b_score"], 0)
            if sa == sb:
                continue
            win_raw = a_id if sa > sb else b_id
            winner = canonicalize(registry, int(season), win_raw)
            if winner not in members:
                continue

            league_name = LEAGUE_FULL.get(lg, lg) if lg else ""

            if cat == "League Championship/Super Bowl":
                # Only call it "Super Bowl ..." from the 1966 season onward.
                if imp == "SB" and int(season) >= 1966:
                    title = _super_bowl_title_for_season(int(season))  # SB 50 -> 'Super Bowl 50'
                else:
                    # Pre-SB era (or non-SB league title in 1966–1969): use league name
                    # e.g., 'NFL Champion', 'AFL Champion', 'AAFC Champion', 'APFA Champion'
                    title = f"{league_name} Champion".strip()
                rows.append({"Season": int(season), "Level": "League", "Title": title})
            else:
                # Conference Championship — skip if CSV already marked this season/team
                if (winner, int(season)) in conf_marks:
                    continue
                clabel = conf_label_map.get((winner, int(season)), "")
                title = f"{clabel} Champion" if clabel else "Conference Champion"
                rows.append({"Season": int(season), "Level": "Conference", "Title": title})

    # ---------- From league_champions.csv (e.g., pre-1933) ----------
    if isinstance(league_champs_df, pd.DataFrame) and not league_champs_df.empty:
        teams_idx = {row["teamID"]: row for _, row in teams_df.iterrows()}

        def _active_in_season(tid: str, s: int) -> bool:
            row = teams_idx.get(tid)
            if row is None: return False
            af = safe_int(row.get("activeFrom"), 9999)
            at_raw = str(row.get("activeTo","")).strip().lower()
            at = 9999 if at_raw == "present" else safe_int(at_raw, 9999)
            return af <= s <= at

        def _resolve_id_by_display(name: str, s: int) -> str | None:
            cand = [tid for tid, row in teams_idx.items() if str(row.get("displayName","")).strip() == name.strip()]
            # prefer one active in that season; else take any
            active = [tid for tid in cand if _active_in_season(tid, s)]
            return (active or cand or [None])[0]

        seen_csv = set()
        for _, r in league_champs_df.iterrows():
            s = _safe_int(r.get("season"))
            if s is None: 
                continue
            lg = norm_league(r.get("league_indicator",""))
            league_name = LEAGUE_FULL.get(lg, lg) if lg else ""

            tid_raw = str(r.get("team_id","")).strip()
            if not tid_raw:
                tdisp = str(r.get("team_display","")).strip()
                if tdisp:
                    tid_raw = _resolve_id_by_display(tdisp, s) or ""
            if not tid_raw:
                continue

            winner = canonicalize(registry, s, tid_raw)
            if winner not in members:
                continue

            title_txt = (str(r.get("title_label","")).strip() 
                         or f"{league_name} Champion").strip()

            # keep one league title per (season, normalized title) from CSV
            key = (int(s), title_txt.casefold())
            if key in seen_csv:
                continue
            seen_csv.add(key)

            rows.append({"Season": int(s), "Level": "League", "Title": title_txt})

    # ---------- Normalize & robust de-dup ----------
    def _norm_title_for_key(level: str, title: str) -> str:
        s = re.sub(r"\s+", " ", str(title)).strip()
        if level in ("Conference", "Division"):
            # strip trailing variations of 'Champion/Championship(s)'
            s = re.sub(r"\b[Cc]hamp(?:ion(?:ship)?)?s?\b\.?$", "", s).strip()
        return s.casefold()

    level_rank = {"Division": 1, "Conference": 2, "League": 3}
    seen = set()
    uniq = []
    for r in sorted(rows, key=lambda x: (int(x["Season"]), level_rank.get(x["Level"], 99), str(x["Title"]))):
        key = (int(r["Season"]), r["Level"], _norm_title_for_key(r["Level"], r["Title"]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    out = pd.DataFrame(uniq, columns=["Season","Level","Title"])
    return out

# -------------------- UI HELPERS --------------------
def next_sequence_for_week(season:int, week_order:int, league:str) -> int:
    with connect_db() as con:
        cur = con.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM games
            WHERE season = ? AND week_order = ? AND league_indicator = ?
        """, (int(season), int(week_order), league))
        count = cur.fetchone()[0] or 0
    return int(count) + 1

# -------------------- RATINGS SUBVIEWS --------------------
def render_table_ui(q_season:int, selected_week_order:int, registry):
    col_show, col_rebuild = st.columns([1, 1])
    if col_show.button("Show ratings", key="tbl_show"):
        df = pretty_ratings_df(q_season, selected_week_order, registry)
        if df.empty:
            st.info("No ratings found for that season/week. Try ‘Recompute ALL ratings’.")
        else:
            st.dataframe(df, use_container_width=True)
    if col_rebuild.button("Recompute ALL ratings", key="tbl_rebuild"):
        with connect_db() as con:
            cur = con.cursor()
            cur.execute("SELECT MIN(season) FROM games;")
            row = cur.fetchone()
        if row and row[0] is not None:
            min_season = int(row[0])
            recompute_ratings_from(min_season, 0, registry)
            _invalidate_cached_dropdowns()
            st.success("Full recompute complete. Now try ‘Show ratings’.")

def render_odds_ui(q_season:int, selected_week_order:int, registry):
    snap = ratings_table(q_season, selected_week_order)
    if snap.empty:
        st.info("No ratings at this snapshot yet.")
        return
    disp_map = registry.get(int(q_season), {}).get("display", {})
    ids_sorted = sorted(snap["team_ID"].tolist(), key=lambda t: disp_map.get(t, t))
    labels = [f"{disp_map.get(t, t)} ({t})" for t in ids_sorted]
    id_of = dict(zip(labels, ids_sorted))
    c1, c2 = st.columns(2)
    pick_a = c1.selectbox("Team A", labels, key="odds_a")
    pick_b = c2.selectbox("Team B", [x for x in labels if x != pick_a], key="odds_b")
    ta = id_of[pick_a]; tb = id_of[pick_b]
    ra = float(snap.loc[snap["team_ID"] == ta, "rating"].iloc[0])
    rb = float(snap.loc[snap["team_ID"] == tb, "rating"].iloc[0])
    dr = ra - rb
    pA = 1.0 / (1.0 + 10 ** (-dr / 600.0))
    pB = 1.0 - pA
    st.write(f"**Win probabilities (neutral):**  {disp_map.get(ta, ta)}: {pA:.1%}  —  {disp_map.get(tb, tb)}: {pB:.1%}")

def render_fullseason_sim_ui(registry, events_csv):
    st.markdown("### Full Season Simulation")
    st.caption(
        "Runs every remaining game for the selected season using your upcoming schedule. "
        "Does not write to the database."
    )

    up = load_upcoming_csv()
    if up.empty:
        st.info("No upcoming schedule found. Add rows in the **Upcoming** tab (games_upcoming.csv).")
        return

    # Ensure helper numeric columns exist and clean
    up = up.copy()
    up["season_int"] = pd.to_numeric(up.get("season"), errors="coerce").astype("Int64")
    up["week_int"]   = pd.to_numeric(up.get("week_order"), errors="coerce").astype("Int64")

    # Seasons = union(actual + upcoming)
    seasons_actual   = set(get_seasons() or [])
    seasons_upcoming = set(int(x) for x in up["season_int"].dropna().astype(int).unique().tolist())
    seasons_all = sorted(seasons_actual | seasons_upcoming)
    if not seasons_all:
        st.info("No seasons available yet.")
        return

    q_season = st.selectbox("Season", seasons_all, index=len(seasons_all)-1, key="fs_season")

    # Baseline week options (ratings if present, else Week 0)
    weeks_have = get_weeks_for_season(q_season)
    if weeks_have:
        week_values = [int(w["week_order"]) for w in weeks_have]
        wk_label = {int(w["week_order"]): str(w["week_label"]) for w in weeks_have}
        base_week_default = max(week_values)
        base_week = st.selectbox(
            "Baseline ratings snapshot",
            week_values,
            index=week_values.index(base_week_default),
            format_func=lambda w: f"{w} — {wk_label.get(int(w),'')}",
            key="fs_base_week",
        )
    else:
        base_week = st.selectbox(
            "Baseline ratings snapshot",
            [0],
            index=0,
            format_func=lambda w: "0 — Week 0 (events, synthesized if needed)",
            key="fs_base_week",
        )
        st.caption("No stored ratings for this season; will synthesize a preseason baseline.")

    # Remaining upcoming games strictly AFTER the baseline week
    rem = up[(up["season_int"] == int(q_season)) & (up["week_int"] > int(base_week))].copy()

    # Clean schedule_df with exactly one numeric week_order
    remaining = rem.drop(columns=["week_order"], errors="ignore").copy()
    remaining["week_order"] = pd.to_numeric(rem["week_int"], errors="coerce").fillna(-1).astype(int)
    remaining = remaining[remaining["week_order"] >= 0]
    # Normalize string columns
    for c in ("team_a_ID","team_b_ID","importance","league_indicator","team_a_display","team_b_display"):
        if c in remaining.columns:
            remaining[c] = remaining[c].astype(str)

    st.caption(f"Simulating **{len(remaining)}** upcoming games (Season {q_season}, weeks > {base_week}).")
    if remaining.empty:
        st.info("No remaining upcoming games after the selected baseline week for this season.")
        return

    # Baseline snapshot ratings (stored or synthesized)
    snap_df = ratings_table(q_season, int(base_week))
    if snap_df.empty:
        base_map = baseline_ratings(int(q_season), int(base_week), registry, events_csv)
        if not base_map:
            st.warning("Couldn't build a baseline (no prior-season ratings to synthesize from).")
            return
        snapshot_ratings = pd.DataFrame([{"team_ID": k, "rating": float(v)} for k, v in base_map.items()])
    else:
        snapshot_ratings = snap_df[["team_ID","rating"]].copy()
        base_map = {r["team_ID"]: float(r["rating"]) for _, r in snapshot_ratings.iterrows()}

    # Controls
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    mode = c1.selectbox("Mode", ["Monte Carlo", "Favorites win"], index=0, key="fs_mode")
    runs = c2.number_input("Runs", min_value=100, max_value=20000, value=1000, step=100,
                           disabled=(mode != "Monte Carlo"), key="fs_runs")
    tie_prob = c3.number_input("Tie probability", min_value=0.0, max_value=0.2, value=0.0, step=0.01,
                               format="%.2f", key="fs_tie")
    seed = c4.number_input("Random seed (optional)", min_value=0, max_value=10_000_000, value=0, step=1, key="fs_seed")

    if st.button("Run full-season simulation", key="fs_go"):
        res = simulate_full_season(
            q_season=int(q_season),
            start_week=int(base_week),
            registry=registry,
            snapshot_ratings=snapshot_ratings,
            schedule_df=remaining,
            mode=mode,
            runs=int(runs),
            tie_prob=float(tie_prob),
            seed=int(seed) if seed else None,
        )

        if res.empty:
            st.warning("Simulation produced no rows. Check your upcoming schedule.")
            return

        # Display: include baseline, expected final mean, and Δ
        disp_map = registry.get(int(q_season), {}).get("display", {})
        res = res.copy()
        res["Team"] = res["team_ID"].map(lambda t: disp_map.get(t, t))
        res["Baseline Rating"] = res["team_ID"].map(lambda t: base_map.get(t, 1000.0))
        res["Exp Final Rating (mean)"] = res["final_rating_mean"].astype(float)
        res["Δ Rating"] = res["Exp Final Rating (mean)"] - res["Baseline Rating"]

        # Current record from actual games up to baseline
        curr = current_record_to_week(int(q_season), int(base_week), registry)
        res["Current W"] = res["team_ID"].map(lambda t: curr.get(t, {}).get("W", 0))
        res["Current L"] = res["team_ID"].map(lambda t: curr.get(t, {}).get("L", 0))
        res["Current T"] = res["team_ID"].map(lambda t: curr.get(t, {}).get("T", 0))

        res["Exp Rem W"] = res["exp_remaining_wins"].astype(float)
        res["Exp Rem T"] = res["exp_remaining_ties"].astype(float)
        res["Exp Final W"] = (res["Current W"] + res["Exp Rem W"]).astype(float)
        res["Exp Final T"] = (res["Current T"] + res["Exp Rem T"]).astype(float)

        show_cols = [
            "Team","Current W","Current L","Current T",
            "Exp Rem W","Exp Rem T","Exp Final W","Exp Final T",
            "Baseline Rating","Exp Final Rating (mean)","Δ Rating",
            "team_ID"
        ]
        out = res[show_cols].sort_values(
            ["Exp Final W","Δ Rating","Exp Final Rating (mean)"],
            ascending=[False, False, False]
        )

        st.dataframe(
            out.style.format({
                "Baseline Rating": "{:.2f}",
                "Exp Final Rating (mean)": "{:.2f}",
                "Δ Rating": "{:+.2f}",
                "Exp Rem W": "{:.2f}",
                "Exp Rem T": "{:.2f}",
                "Exp Final W": "{:.2f}",
                "Exp Final T": "{:.2f}",
            }),
            use_container_width=True
        )

        # Sanity / debug summary
        with st.expander("Simulation summary & checks", expanded=False):
            n_games = len(remaining)
            total_exp_wins = float(res["Exp Rem W"].sum()) if "Exp Rem W" in res.columns else 0.0
            st.write(f"Games simulated: **{n_games}**")
            st.write(f"Sum of expected remaining wins across all teams: **{total_exp_wins:.2f}** "
                     f"(should be close to number of games; exactly equal if no ties)")
            if tie_prob == 0:
                st.caption("With tie_prob = 0, the sum of expected wins should equal the game count.")

def render_charts_ui(q_season:int, selected_week_order:int, registry, events_csv:pd.DataFrame, teams_csv:pd.DataFrame):
    with st.expander("Chart controls", expanded=True):
        fr_map, id2fr, fr_labels = build_franchise_index(events_csv, teams_csv)
        def franchise_terminal_options() -> dict[str, str]:
            teams_idx = {row["teamID"]: row for _, row in teams_csv.iterrows()}
            old_ids = set()
            if not events_csv.empty:
                ren = events_csv[events_csv["type"] == "RENAME"]
                for _, ev in ren.iterrows():
                    for t in [x.strip() for x in str(ev.get("teamIDs","")).split(",") if x.strip()]:
                        old_ids.add(t)
            def active_to_year(tid: str) -> int:
                row = teams_idx.get(tid)
                if row is None: return -1
                s = str(row.get("activeTo", "")).strip().lower()
                if s == "present": return 9999
                try: return int(s)
                except: return -1
            def disp_label(tid: str) -> str:
                row = teams_idx.get(tid)
                return str(row.get("displayName", tid)) if row is not None else str(tid)
            opts: dict[str, str] = {}
            for root, members in fr_map.items():
                members = list(members)
                cands = [t for t in members if t not in old_ids] or members
                term = max(cands, key=active_to_year)
                label = f"{disp_label(term)} (franchise)"
                opts[label] = root
            return opts
        autoscale = st.checkbox("Autoscale y-axis", value=False, key="y_autoscale")
        col_y1, col_y2 = st.columns(2)
        y_min = col_y1.number_input("Y min", value=600, step=10, key="y_min")
        y_max = col_y2.number_input("Y max", value=1400, step=10, key="y_max")
        y_scale = None if autoscale else alt.Scale(domain=[float(y_min), float(y_max)])
        show_points = st.checkbox("Show points at each week", value=True, key="show_points")
        multi_year = st.checkbox("Multi-year mode", value=False, key="multi_year")
        franchise_mode = st.checkbox("Franchise mode (group eras & shared merge teams)", value=False, key="fr_mode")
        def explode_by_franchise(df_in: pd.DataFrame, selected_roots: set[str], series_col: str) -> pd.DataFrame:
            rows = []; sel = set(selected_roots)
            for _, r in df_in.iterrows():
                frs = id2fr.get(r["team_ID"], set())
                tag = frs & sel
                if not tag: continue
                for root in tag:
                    rec = r.copy(); rec[series_col] = fr_labels.get(root, root); rows.append(rec)
            return pd.DataFrame(rows)

    if not multi_year:
        season_df = ratings_long_df(q_season, registry)
        if season_df.empty: st.info("No ratings to chart for this season yet."); return
        if franchise_mode:
            term_opts = franchise_terminal_options()
            all_fr = sorted(term_opts.keys())
            picked_fr_labels = st.multiselect("Franchises to plot", all_fr, default=[], key="single_fr_pick")
            picked_roots = {term_opts[x] for x in picked_fr_labels}
            plot_df = explode_by_franchise(season_df, picked_roots, series_col="Series")
            if plot_df.empty: st.info("Pick at least one franchise."); return
            x_zoom = st.checkbox("Enable X-axis zoom/pan (simpler x labels)", value=True, key="single_xzoom_fr")
            x_enc = alt.X("t:Q", title="Week", axis=alt.Axis(tickMinStep=1)) if x_zoom else alt.X("XLabel:N", title="Season • Week", sort=alt.SortField(field="t", order="ascending"))
            chart = line_points_chart(plot_df, x_enc, "Series", "Franchise", y_scale, "t",
                                      [alt.Tooltip("Series:N", title="Franchise"), alt.Tooltip("Team:N", title="Team (era)"),
                                       alt.Tooltip("season:Q", title="Season"), alt.Tooltip("week:Q", title="Week"),
                                       alt.Tooltip("WeekLabel:N", title="Label"), alt.Tooltip("rating:Q", format=".2f", title="Rating")],
                                      height=360, show_points=show_points)
            st.altair_chart(chart, use_container_width=True)
        else:
            all_teams = sorted(season_df["Team"].unique().tolist())
            picked = st.multiselect("Teams to plot (eras)", all_teams, default=[], key="single_team_select")
            plot_df = season_df if not picked else season_df[season_df["Team"].isin(picked)]
            if plot_df.empty: st.info("Pick at least one team to display a chart."); return
            x_zoom = st.checkbox("Enable X-axis zoom/pan (simpler x labels)", value=True, key="single_xzoom")
            x_enc = alt.X("t:Q", title="Week", axis=alt.Axis(tickMinStep=1)) if x_zoom else alt.X("XLabel:N", title="Season • Week", sort=alt.SortField(field="t", order="ascending"))
            chart = line_points_chart(plot_df, x_enc, "Team", "Team", y_scale, "t",
                                      [alt.Tooltip("Team:N", title="Team"), alt.Tooltip("season:Q", title="Season"),
                                       alt.Tooltip("week:Q", title="Week"), alt.Tooltip("WeekLabel:N", title="Label"),
                                       alt.Tooltip("rating:Q", format=".2f", title="Rating")],
                                      height=360, show_points=show_points)
            st.altair_chart(chart, use_container_width=True)
    else:
        seasons_all = get_seasons()
        if not seasons_all: st.info("No seasons available."); return
        smin_all, smax_all = min(seasons_all), max(seasons_all)
        default_start = max(smin_all, smax_all - 9)
        s_min, s_max = st.slider("Season range", min_value=smin_all, max_value=smax_all,
                                 value=(default_start, smax_all), step=1, key="season_range")
        multi_df = ratings_long_df_range(s_min, s_max, registry)
        if multi_df.empty: st.info("No ratings in that range."); return
        if franchise_mode:
            term_opts = franchise_terminal_options()
            all_fr = sorted(term_opts.keys())
            picked_fr_labels = st.multiselect("Franchises to plot", all_fr, default=[], key="multi_fr_pick")
            picked_roots = {term_opts[x] for x in picked_fr_labels}
            plot_df = explode_by_franchise(multi_df, picked_roots, series_col="Series")
            if plot_df.empty: st.info("Pick at least one franchise."); return
            time_ix = build_timeline_index(s_min, s_max)
            plot_df = plot_df.merge(time_ix, on=["season", "week"], how="left")
            show_separators = st.checkbox("Show season separators", value=True, key="sep_fr")
            seps = season_boundaries_df(time_ix) if show_separators else pd.DataFrame()
            x_zoom_multi = st.checkbox("Enable X-axis zoom/pan (simpler x labels)", value=True, key="multi_xzoom_fr")
            x_enc = alt.X("t_cont:Q", title="Season • Week", axis=alt.Axis(tickMinStep=1)) if x_zoom_multi else alt.X("XLabel:N", title="Season • Week", sort=alt.SortField(field="t", order="ascending"))
            chart = line_points_chart(plot_df, x_enc, "Series", "Franchise",
                                      None if st.session_state.get("y_autoscale") else alt.Scale(domain=[float(st.session_state.get("y_min",600)), float(st.session_state.get("y_max",1400))]),
                                      "t",
                                      [alt.Tooltip("Series:N", title="Franchise"), alt.Tooltip("Team:N", title="Team (era)"),
                                       alt.Tooltip("season:Q", title="Season"), alt.Tooltip("week:Q", title="Week"),
                                       alt.Tooltip("WeekLabel:N", title="Label"), alt.Tooltip("rating:Q", format=".2f", title="Rating")],
                                      height=380, show_points=st.session_state.get("show_points", True))
            if show_separators and not seps.empty:
                rules = alt.Chart(seps).mark_rule(opacity=0.25).encode(x="t_cont:Q")
                chart = alt.layer(chart, rules).resolve_scale(x="shared")
            st.altair_chart(chart, use_container_width=True)
        else:
            all_opts = sorted(multi_df["Legend"].unique().tolist())
            last_week = multi_df[multi_df["season"] == s_max]["week"].max()
            tail = multi_df[(multi_df["season"] == s_max) & (multi_df["week"] == last_week)]
            top_default = tail.sort_values("rating", ascending=False)["Legend"].head(3).tolist()
            picked_multi = st.multiselect("Teams to plot (eras)", all_opts, default=top_default, key="multi_team_select")
            plot_df = multi_df if not picked_multi else multi_df[multi_df["Legend"].isin(picked_multi)]
            if plot_df.empty: st.info("Pick at least one team to display a chart."); return
            time_ix = build_timeline_index(s_min, s_max)
            plot_df = plot_df.merge(time_ix, on=["season", "week"], how="left")
            show_separators = st.checkbox("Show season separators", value=True, key="sep_era")
            seps = season_boundaries_df(time_ix) if show_separators else pd.DataFrame()
            x_zoom_multi = st.checkbox("Enable X-axis zoom/pan (simpler x labels)", value=True, key="multi_xzoom")
            x_enc = alt.X("t_cont:Q", title="Season • Week", axis=alt.Axis(tickMinStep=1)) if x_zoom_multi else alt.X("XLabel:N", title="Season • Week", sort=alt.SortField(field="t", order="ascending"))
            chart = line_points_chart(plot_df, x_enc, "Legend", "Team",
                                      None if st.session_state.get("y_autoscale") else alt.Scale(domain=[float(st.session_state.get("y_min",600)), float(st.session_state.get("y_max",1400))]),
                                      "t",
                                      [alt.Tooltip("Legend:N", title="Team"), alt.Tooltip("season:Q", title="Season"),
                                       alt.Tooltip("week:Q", title="Week"), alt.Tooltip("WeekLabel:N", title="Label"),
                                       alt.Tooltip("rating:Q", format=".2f", title="Rating")],
                                      height=380, show_points=st.session_state.get("show_points", True))
            if show_separators and not seps.empty:
                rules = alt.Chart(seps).mark_rule(opacity=0.25).encode(x="t_cont:Q")
                chart = alt.layer(chart, rules).resolve_scale(x="shared")
            st.altair_chart(chart, use_container_width=True)

# -------------------- STREAMLIT APP --------------------
st.set_page_config(page_title="NFL Historical Rankings", layout="wide")
st.title("NFL Historical Rankings (Streamlit)")

games_csv, events_csv, teams_csv = load_csvs()
divisions_csv = load_divisions_csv()
league_champs_df = load_league_champions_csv()
init_db()
mirror_into_db(games_csv)
registry = build_season_registry(events_csv, teams_csv)

# Build ratings if empty (no upcoming interference)
if ratings_is_empty():
    with connect_db() as con:
        cur = con.cursor()
        cur.execute("SELECT MIN(season) FROM games;")
        row = cur.fetchone()
    if row and row[0] is not None:
        min_season = int(row[0])
        recompute_ratings_from(min_season, 0, registry)
        _invalidate_cached_dropdowns()

# Load upcoming fixtures (separate file)
upcoming_df = load_upcoming_csv()

tabs = st.tabs(["Ratings", "Add Game", "Games", "Events", "Teams", "Upcoming", "Simulator", "Franchise Stats", "Trophy Case"])

# ---------- Add Game (real) ----------
with tabs[1]:
    st.subheader("Add a new game (real)")
    col1, col2 = st.columns(2)
    league = col1.selectbox("League", sorted(list(LEAGUES)))
    season = col2.number_input("Season", min_value=1920, max_value=2100, value=dt.date.today().year)
    week_order = st.number_input("Week (numeric order; 0..40 incl. postseason rounds)", min_value=0, max_value=40, value=1)
    week_label_input = st.text_input("Week label (display only; e.g., 15 or 'NFL Championship')", value=str(int(week_order)))
    seq_next = next_sequence_for_week(int(season), int(week_order), league)
    gid = f"S{int(season):04X}W{int(week_order):02X}G{int(seq_next):02X}{league}"
    imp = st.selectbox("Importance", ["NCR","CR","DR","WC","DV","CC","SB"])
    active_ids = sorted(list(registry.get(int(season), {"active": set()})["active"]))
    team_a = st.selectbox("Team A (ID)", active_ids)
    team_b = st.selectbox("Team B (ID)", active_ids)
    score_a = st.number_input("Score A", min_value=0, max_value=200, value=20)
    score_b = st.number_input("Score B", min_value=0, max_value=200, value=17)
    disp_map = registry.get(int(season), {}).get("display", {})
    team_a_display = disp_map.get(team_a, team_a)
    team_b_display = disp_map.get(team_b, team_b)

    if st.button("Save game & recompute"):
        new_row = {
            "league_indicator": league, "season": int(season), "week": week_label_input, "game_ID": gid,
            "importance": imp, "team_a_display": team_a_display, "team_a_ID": team_a, "team_a_score": int(score_a),
            "team_b_display": team_b_display, "team_b_ID": team_b, "team_b_score": int(score_b),
        }
        parsed = parse_game_id(new_row["game_ID"])
        if not parsed:
            st.error("game_ID format invalid.")
        elif parsed["league_indicator"] != new_row["league_indicator"]:
            st.error("game_ID league mismatch.")
        elif new_row["importance"] not in IMPORTANCE_K:
            st.error("importance invalid.")
        elif new_row["team_a_ID"] == new_row["team_b_ID"]:
            st.error("Teams must differ.")
        else:
            games_csv = pd.concat([games_csv, pd.DataFrame([new_row])], ignore_index=True)
            games_csv.to_csv(DATA_DIR / "games.csv", index=False)
            mirror_into_db(games_csv)
            recompute_ratings_from(int(season), int(week_order), registry)
            _invalidate_cached_dropdowns()
            st.success(f"Saved {gid} and recomputed ratings from {season}, week {week_order} onward.")
            try:
                gh_sync_to_repo(
                    {
                        "data/games.csv": DATA_DIR / "games.csv",
                        # include upcoming if you want it tracked even when unchanged
                        "data/games_upcoming.csv": UPCOMING_CSV if UPCOMING_CSV.exists() else DATA_DIR / "games_upcoming.csv",
                    },
                    message=f"Add game {gid} ({team_a_display} {score_a}–{score_b} {team_b_display})"
                )
                st.success("Synced data back to GitHub.")
            except Exception as e:
                st.warning(f"Couldn’t push CSVs to GitHub: {e}")
# ---------- Games (real list) ----------
with tabs[2]:
    st.subheader("Games (actual)")
    st.dataframe(read_games_db(), use_container_width=True)

# ---------- Events ----------
with tabs[3]:
    st.subheader("EVENT LOGS (Week 0, applied in order)")
    st.caption("Supported types: LEAVE, UNMERGE, MERGE, SWITCH, REJOIN, RENAME, JOIN. Optional columns: league, fromLeague, toLeague.")
    st.dataframe(events_csv, use_container_width=True)

# ---------- Teams ----------
with tabs[4]:
    st.subheader("TEAM LOG")
    st.dataframe(teams_csv, use_container_width=True)

# ---------- Ratings ----------
with tabs[0]:
    st.subheader("Ratings")
    seasons = get_seasons()
    if not seasons:
        st.info("No games/ratings found. Add games or load CSVs, then recompute.")
    else:
        default_season = seasons[-1]
        season_idx = seasons.index(default_season) if default_season in seasons else 0
        q_season = st.selectbox("Season", seasons, index=season_idx)
        weeks = get_weeks_for_season(q_season)
        if not weeks:
            st.info("This season has no ratings yet. Try ‘Recompute ALL ratings’.")
        else:
            week_options, week_values = [], []
            for w in weeks:
                wo = int(w["week_order"]); wl = str(w["week_label"])
                label = f"{wo} — {wl}" if wl and wl != str(wo) else f"{wo}"
                week_options.append(label); week_values.append(wo)
            week_idx = len(week_values) - 1
            selected_label = st.selectbox("Week", week_options, index=week_idx)
            selected_week_order = week_values[week_options.index(selected_label)]
            subtab = st.selectbox("Section", ["Table", "Charts", "Odds"], index=0, key="ratings_section")
            if subtab == "Table":
                render_table_ui(q_season, selected_week_order, registry)
            elif subtab == "Charts":
                render_charts_ui(q_season, selected_week_order, registry, events_csv, teams_csv)
            elif subtab == "Odds":
                render_odds_ui(q_season, selected_week_order, registry)

# ---------- Upcoming (separate; does NOT affect ratings) ----------
with tabs[5]:
    st.subheader("Upcoming (does not affect rankings)")
    up_df = load_upcoming_csv()
    st.write("**Current upcoming games**")
    st.dataframe(up_df, use_container_width=True)

    with st.expander("Add upcoming game"):
        col1, col2 = st.columns(2)
        lg = col1.selectbox("League", sorted(list(LEAGUES)), key="up_league")
        up_season = col2.number_input("Season", min_value=1920, max_value=2100, value=dt.date.today().year, key="up_season")
        up_week = st.number_input("Week (numeric order)", min_value=0, max_value=40, value=1, key="up_week")
        up_label = st.text_input("Week label (display only)", value=str(int(up_week)), key="up_label")
        up_imp = st.selectbox("Importance", ["NCR","CR","DR","WC","DV","CC","SB"], index=1, key="up_imp")
        active_ids = sorted(list(registry.get(int(up_season), {"active": set()})["active"]))
        colA, colB = st.columns(2)
        up_a = colA.selectbox("Team A (ID)", active_ids, key="up_a")
        up_b = colB.selectbox("Team B (ID)", active_ids, key="up_b")
        disp_map_u = registry.get(int(up_season), {}).get("display", {})
        up_a_disp = disp_map_u.get(up_a, up_a)
        up_b_disp = disp_map_u.get(up_b, up_b)

        if st.button("Save upcoming game", key="save_upcoming"):
            if up_a == up_b:
                st.error("Teams must differ.")
            else:
                cur_up = load_upcoming_csv()
                uid = next_upcoming_id(cur_up)
                newu = pd.DataFrame([{
                    "upcoming_ID": uid, "league_indicator": lg, "season": int(up_season),
                    "week_order": int(up_week), "week_label": up_label, "importance": up_imp,
                    "team_a_display": up_a_disp, "team_a_ID": up_a,
                    "team_b_display": up_b_disp, "team_b_ID": up_b
                }])
                cur_up = pd.concat([cur_up, newu], ignore_index=True)
                save_upcoming_csv(cur_up)
                st.success(f"Added upcoming game #{uid}: {up_a_disp} vs {up_b_disp} (Season {up_season}, Week {up_week}).")

    with st.expander("Promote an upcoming game to actual (enter final score)"):
        cur_up = load_upcoming_csv()
        if cur_up.empty:
            st.info("No upcoming games to promote.")
        else:
            disp = []
            for _, r in cur_up.iterrows():
                disp.append(f"#{int(r['upcoming_ID'])} • {int(r['season'])} W{int(r['week_order'])} • {r['team_a_display']} vs {r['team_b_display']} ({r['league_indicator']})")
            picked = st.selectbox("Pick an upcoming game", disp, key="prom_pick")
            idx = disp.index(picked)
            row = cur_up.iloc[idx]
            s_season = int(row["season"]); s_week = int(row["week_order"]); s_league = str(row["league_indicator"])
            seq_next = next_sequence_for_week(s_season, s_week, s_league)
            gid = f"S{s_season:04X}W{s_week:02X}G{seq_next:02X}{s_league}"
            st.caption(f"Assigned game_ID on promotion: `{gid}`")
            colsa, colsb = st.columns(2)
            sc_a = colsa.number_input(f"Final score: {row['team_a_display']}", min_value=0, max_value=200, value=20, key="prom_sa")
            sc_b = colsb.number_input(f"Final score: {row['team_b_display']}", min_value=0, max_value=200, value=17, key="prom_sb")

            if st.button("Promote to actual & recompute", key="do_promote"):
                new_real = {
                    "league_indicator": s_league,
                    "season": s_season,
                    "week": str(row["week_label"]),
                    "game_ID": gid,
                    "importance": str(row["importance"]),
                    "team_a_display": str(row["team_a_display"]),
                    "team_a_ID": str(row["team_a_ID"]),
                    "team_a_score": int(sc_a),
                    "team_b_display": str(row["team_b_display"]),
                    "team_b_ID": str(row["team_b_ID"]),
                    "team_b_score": int(sc_b),
                }
                games_csv = pd.concat([games_csv, pd.DataFrame([new_real])], ignore_index=True)
                games_csv.to_csv(DATA_DIR / "games.csv", index=False)
                mirror_into_db(games_csv)
                cur_up = cur_up.drop(cur_up.index[idx]).reset_index(drop=True)
                save_upcoming_csv(cur_up)
                recompute_ratings_from(s_season, s_week, registry)
                _invalidate_cached_dropdowns()
                st.success(f"Promoted {gid} and recomputed from {s_season}, week {s_week} onward.")
                try:
                    gh_sync_to_repo(
                        {
                            "data/games.csv": DATA_DIR / "games.csv",
                            "data/games_upcoming.csv": UPCOMING_CSV,
                        },
                        message=f"Promote upcoming -> actual: {gid} "
                                f"({row['team_a_display']} {int(sc_a)}–{int(sc_b)} {row['team_b_display']})"
                    )
                    st.success("Synced data back to GitHub.")
                except Exception as e:
                    st.warning(f"Couldn’t push CSVs to GitHub: {e}")
# ---------- Simulator (what-if; does NOT save) ----------
with tabs[6]:
    st.subheader("Simulator (what-if; no changes are saved)")
    up_all = load_upcoming_csv()
    if up_all.empty:
        st.info("Add some games in the **Upcoming** tab first.")
    else:
        seasons_up = sorted({int(x) for x in up_all["season"].dropna().astype(int).unique().tolist()})
        sim_season = st.selectbox("Upcoming Season", seasons_up, index=len(seasons_up)-1)
        weeks_up = sorted({int(x) for x in up_all[up_all["season"]==sim_season]["week_order"].dropna().astype(int).unique().tolist()})
        sim_week = st.selectbox("Upcoming Week (numeric)", weeks_up, index=0)

        weeks_have = get_weeks_for_season(sim_season)
        if weeks_have:
            week_vals = [int(w["week_order"]) for w in weeks_have]
            base_week_default = max(week_vals)
            base_label = {int(w["week_order"]): w["week_label"] for w in weeks_have}
            sim_base_week = st.selectbox("Baseline ratings snapshot", week_vals, index=week_vals.index(base_week_default),
                                         format_func=lambda w: f"{w} — {base_label.get(int(w),'')}")
        else:
            sim_base_week = 0
            st.caption("No stored ratings for this season; using synthesized preseason (Week 0).")

        fixtures = up_all[(up_all["season"]==sim_season) & (up_all["week_order"]==sim_week)].copy()
        if fixtures.empty:
            st.info("No upcoming games for that season/week.")
        else:
            # Baseline ratings
            
            base_r = baseline_ratings(sim_season, sim_base_week, registry, events_csv)
            if not base_r:
                st.warning("Could not build a baseline ratings snapshot for this season.")
            else:
                st.write("### Pick winners (leave as ‘Auto by probability’ to randomize)")
                disp_map = registry.get(int(sim_season), {}).get("display", {})
                forced = []
                for i, (_, g) in enumerate(fixtures.iterrows()):
                    ra = base_r.get(canonicalize(registry, sim_season, g["team_a_ID"]), 1000.0)
                    rb = base_r.get(canonicalize(registry, sim_season, g["team_b_ID"]), 1000.0)
                    pA = 1.0 / (1.0 + 10 ** (-(ra-rb)/600.0))
                    label = f"{g['team_a_display']} vs {g['team_b_display']} • {g['league_indicator']} • {g['importance']}  —  P({g['team_a_display']}) ≈ {pA:.1%}"
                    pick = st.radio(label, ["Auto by probability", g["team_a_display"], g["team_b_display"]],
                                    index=0, key=f"sim_pick_{int(g['upcoming_ID'])}")
                    fw = "A" if pick == g["team_a_display"] else "B" if pick == g["team_b_display"] else ""
                    forced.append(fw)
                fixtures = fixtures.copy()
                fixtures["forced_winner"] = forced

                def simulate_once(start_ratings: Dict[str,float], fixtures: pd.DataFrame, registry, season:int) -> Dict[str,float]:
                    r = start_ratings.copy()
                    for _, g in fixtures.iterrows():
                        a = canonicalize(registry, season, str(g["team_a_ID"]))
                        b = canonicalize(registry, season, str(g["team_b_ID"]))
                        if not a or not b or a == INDEPENDENT_ID or b == INDEPENDENT_ID: continue
                        ra = r.get(a, 1000.0); rb = r.get(b, 1000.0)
                        imp = IMPORTANCE_K.get(str(g["importance"]).strip(), 15)
                        dr = ra - rb
                        pA = 1.0 / (1.0 + 10 ** (-(dr) / 600.0))
                        choice = str(g.get("forced_winner", "")).strip()
                        if choice == "A":
                            sa, sb = 1, 0
                        elif choice == "B":
                            sa, sb = 0, 1
                        else:
                            sa, sb = (1,0) if random.random() < pA else (0,1)
                        ra_new, rb_new = elo_update(ra, rb, sa, sb, imp)
                        r[a], r[b] = ra_new, rb_new
                    return r

                if st.button("Run one simulation"):
                    sim_r = simulate_once(base_r, fixtures, registry, sim_season)
                    rows = []
                    for tid in set(list(sim_r.keys()) + list(base_r.keys())):
                        r0 = base_r.get(tid, 1000.0); r1 = sim_r.get(tid, 1000.0)
                        if abs(r1 - r0) < 1e-9: continue
                        rows.append({"team_ID": tid, "Start": r0, "Sim": r1, "Δ": r1 - r0,
                                     "Team": disp_map.get(tid, tid)})
                    out = pd.DataFrame(rows).sort_values("Δ", ascending=False)
                    if out.empty:
                        st.info("No rating changes (check baseline and fixtures).")
                    else:
                        st.write("### Simulated rating changes (single pass)")
                        st.dataframe(out[["Team","Start","Sim","Δ"]].style.format({"Start":"{:.2f}","Sim":"{:.2f}","Δ":"{:+.2f}"}),
                                     use_container_width=True)
                        st.caption("This is a what-if preview only. Nothing was written to the database.")

    st.divider()
    render_fullseason_sim_ui(registry, events_csv)

# ---------- Franchise Stats ----------
with tabs[7]:
    st.subheader("Franchise Stats")

    # build franchise options (terminal names)
    fr_map, _, fr_labels = build_franchise_index(events_csv, teams_csv)

    def franchise_terminal_options_for_tab() -> dict[str, str]:
        teams_idx = {row["teamID"]: row for _, row in teams_csv.iterrows()}

        # IDs that were renamed (prefer terminal identity for display)
        old_ids = set()
        if not events_csv.empty:
            ren = events_csv[events_csv["type"] == "RENAME"]
            for _, ev in ren.iterrows():
                for t in [x.strip() for x in str(ev.get("teamIDs","")).split(",") if x.strip()]:
                    old_ids.add(t)

        def to_year(v):
            s = str(v).strip().lower()
            if s == "present": return 9999
            try: return int(s)
            except: return None

        def disp_label(tid: str) -> str:
            row = teams_idx.get(tid)
            return str(row.get("displayName", tid)) if row is not None else str(tid)

        def span_for_franchise(members: set[str]):
            starts, ends = [], []
            for t in members:
                row = teams_idx.get(t, None)
                if row is None:
                    continue  # <-- fix: don't try truth-test a Series
                y1 = to_year(row.get("activeFrom", ""))
                y2 = to_year(row.get("activeTo",   ""))
                if y1 is not None: starts.append(y1)
                if y2 is not None: ends.append(y2)
            if not starts or not ends:
                return None, None
            return min(starts), max(ends)

        def _end_year(tid: str) -> int:
            row = teams_idx.get(tid)
            y = to_year(row.get("activeTo", "")) if row is not None else None
            return -1 if y is None else y
 
        opts: dict[str, str] = {}
        for root, members in fr_map.items():
            mems = list(members)
            cands = [t for t in mems if t not in old_ids] or mems
            term = max(cands, key=_end_year)

            start, end = span_for_franchise(members)
            start_s = "—" if start is None else str(start)
            end_s   = "Present" if end == 9999 else ("—" if end is None else str(end))

            label = f"{disp_label(term)} ({start_s}–{end_s})"
            opts[label] = root

        return opts



    term_opts = franchise_terminal_options_for_tab()
    if not term_opts:
        st.info("No franchise data available.")
    else:
        pick = st.selectbox("Choose franchise", sorted(term_opts.keys()))
        fr_root = term_opts[pick]

        stats = compute_franchise_stats(fr_root, events_csv, teams_csv, registry, divisions_csv)

        colA, colB = st.columns(2)
        # Best / Worst with ratings and context
        if stats.get("best_when") and stats.get("best_rank") not in (None, 0):
            s, w, lbl, tname = stats["best_when"]
            colA.markdown(f"**Best Ranking:** #{stats['best_rank']}  \nRating: `{stats['best_rank_rating']:.2f}`  \nWhen: **{s} {lbl}** ({tname})")
        else:
            colA.markdown("**Best Ranking:** —")
        if stats.get("worst_when") and stats.get("worst_rank") not in (None, 0):
            s2, w2, lbl2, tname2 = stats["worst_when"]
            colB.markdown(f"**Worst Ranking:** #{stats['worst_rank']}  \nRating: `{stats['worst_rank_rating']:.2f}`  \nWhen: **{s2} {lbl2}** ({tname2})")
        else:
            colB.markdown("**Worst Ranking:** —")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Weeks at #1", f"{stats.get('weeks_at_1',0)}")
        col2.metric("Longest #1 streak", f"{stats.get('longest_1_streak',0)}")
        col3.metric("Weeks in Top 5", f"{stats.get('weeks_top5',0)}")
        col4.metric("Longest Top 5 streak", f"{stats.get('longest_top5_streak',0)}")

        # --- Name history ---
        st.markdown("### Name history")
        names = stats.get("names_history", stats.get("name_history", []))
        if names:
            nh = pd.DataFrame(names)
            def fmt_span(r):
                f = "—" if r["From"] in (None, -1) else str(int(r["From"]))
                t = "Present" if r["To"] == 9999 else "—" if r["To"] in (None, -1) else str(int(r["To"]))
                return f"{f}–{t}"
            nh["Years"] = nh.apply(fmt_span, axis=1)
            nh = nh[["Name","Years"]]
            st.dataframe(nh, hide_index=True, use_container_width=True)
        else:
            st.info("No name history found for this franchise.")

        # --- Leagues participated ---
        st.markdown("### Leagues participated")
        leagues_full = stats.get("leagues_participated", stats.get("leagues", []))
        if leagues_full:
            st.write(", ".join(sorted(leagues_full)))
        else:
            st.write("—")

        # --- Titles (MOVED here, under Leagues participated) ---
        st.markdown("### Titles")
        titles_df = franchise_titles_table(fr_root, divisions_csv, events_csv, teams_csv, registry, league_champs_df)

        if titles_df.empty:
            st.caption("No titles found for this franchise yet.")
        else:
            c1, c2, c3, _ = st.columns([20, 20, 20, 3])
            counts = titles_df["Level"].value_counts()
            c1.metric("League / Super Bowl", int(counts.get("League", 0)))
            c2.metric("Conference", int(counts.get("Conference", 0)))
            c3.metric("Division", int(counts.get("Division", 0)))

        st.dataframe(
            titles_df[["Season", "Title", "Level"]],
            hide_index=True,
            use_container_width=True
        )

        # --- Playoff wins (unchanged) ---
        st.markdown("### Playoff wins")
        pw = stats.get("playoff_wins_df")
        if isinstance(pw, pd.DataFrame) and not pw.empty:
            st.dataframe(pw[["Season","Round","Category","Opponent","Score","League"]],
                         use_container_width=True, hide_index=True)
            counts = stats.get("playoff_counts", {})
            if counts:
                st.caption("Wins by round:")
                cnt_df = pd.DataFrame([{"Round": k, "Wins": v} for k, v in counts.items()]).sort_values("Wins", ascending=False)
                st.dataframe(cnt_df, use_container_width=False, hide_index=True)
        else:
            st.info("No playoff wins found for this franchise.")

# ---------- Trophy Case (NFL Cups) ----------
with tabs[8]:
    st.subheader("Trophy Case (NFL Cups)")
    st.caption("Holder = better all-time NFL record; ties broken by point differential.")

    # Choose season (to know the current divisions) -> pick latest NFL season in divisions.csv
    nfl_div_seasons = divisions_csv[divisions_csv["league_indicator"].map(norm_league) == "N"]["season"].dropna()
    nfl_div_seasons = pd.to_numeric(nfl_div_seasons, errors="coerce").dropna().astype(int).tolist()
    if not nfl_div_seasons:
        st.info("No NFL divisions found. Add divisions.csv rows for the season you want.")
    else:
        default_season = max(nfl_div_seasons)
        q_season = st.number_input("Season (for divisions layout)", min_value=min(nfl_div_seasons),
                                   max_value=max(nfl_div_seasons), value=default_season, step=1)

        divisions, root_to_teamid, root_to_label = active_nfl_divisions_for_season(
            int(q_season), divisions_csv, events_csv, teams_csv, registry
        )
        if not divisions:
            st.info(f"No NFL division data for {q_season}.")
        else:
            # Build franchise pick list from active NFL teams this season
            active_roots = []
            active_labels = []
            for d in divisions:
                for r in d["roots"]:
                    active_roots.append(r)
                    active_labels.append(root_to_label.get(r, r))
            # Dedup while preserving order
            seen = set(); opts = []
            for r, lab in zip(active_roots, active_labels):
                if r not in seen:
                    opts.append((lab, r)); seen.add(r)
            labels_only = [x[0] for x in opts]
            pick = st.selectbox("Franchise", labels_only, index=0)
            selected_root = dict(opts)[pick]

            # Identify the user's division row and put it first
            my_div_idx = None
            for i, d in enumerate(divisions):
                if selected_root in d["roots"]:
                    my_div_idx = i
                    break
            ordered = []
            if my_div_idx is not None:
                # top row: my division (remove me -> 3 opponents)
                mine = divisions[my_div_idx].copy()
                roots_no_self = [r for r in mine["roots"] if r != selected_root]
                mine["roots"] = roots_no_self
                ordered.append(mine)
                # the rest below
                ordered.extend([d for j, d in enumerate(divisions) if j != my_div_idx])
            else:
                ordered = divisions

            # Compute H2H vs all active opponents once
            opp_set = set([r for d in ordered for r in d["roots"]])
            h2h = h2h_map_for_selected_franchise(selected_root, opp_set, events_csv, teams_csv)

            # Optional cup names
            cups_df = load_cups_csv()
            cup_name_map = {}
            for _, r in cups_df.iterrows():
                a = str(r.get("franchise_a","")).strip()
                b = str(r.get("franchise_b","")).strip()
                n = str(r.get("cup_name","")).strip()
                if a and b and n:
                    cup_name_map[(tuple(sorted([a,b])))] = n

            def cup_name(a_root: str, b_root: str) -> str:
                k = tuple(sorted([a_root, b_root]))
                if k in cup_name_map:
                    return cup_name_map[k]
                # fallback auto-name from this season's labels
                la = root_to_label.get(a_root, a_root)
                lb = root_to_label.get(b_root, b_root)
                return f"{la} – {lb} Cup"

            # Render rows: top (3) then the rest (4 each)
            for row_i, d in enumerate(ordered):
                conf, div = d["conf"], d["div"]
                row_roots = d["roots"]
                ncols = len(row_roots)  # 3 for my division; otherwise usually 4
                if ncols == 0:
                    continue
                st.markdown(f"#### {conf} {div}")
                cols = st.columns(ncols)
                for j, opp_root in enumerate(row_roots):
                    rec = h2h.get(opp_root, {"W":0,"L":0,"T":0,"PD":0})
                    has = _holder_is_selected(rec)
                    opp_label = root_to_label.get(opp_root, opp_root)
                    name = cup_name(selected_root, opp_root)
                    owned_txt = "🏆 **Owned**" if has is True else ("— **Tied (no holder)**" if has is None else "❌ **Not owned**")
                    with cols[j]:
                        st.markdown(
                            f"{owned_txt}  \n"
                            f"**{name}**  \n"
                            f"vs **{opp_label}**  \n"
                            f"Record: `{_format_record(rec)}`"
                        )

            st.caption("Rule: Holder = best all-time NFL W-L in the series; if tied, higher all-time point differential holds. Perfect ties = no holder.")

