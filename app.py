"""
Manufacturing Trial Tracker — Streamlit + SQLite app.

Tabs:
  1. Data Entry  — log a new trial with clickable grid cell picker and dynamic fields
  2. Dashboard   — heat map, bar chart, trial log with filters
  3. Analysis    — correlations, worst grid cells, summary stats

DB is seeded from CSV on first run if the trials table is empty.
"""

import re
import sqlite3
import os
from pathlib import Path
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DB_PATH  = BASE_DIR / "trials.db"
CSV_PATH = Path(r"C:\Users\Tsion\Downloads\Tension and Airflow - Sheet1.csv")

# ── constants ─────────────────────────────────────────────────────────────────
GRID_ROWS = ["A", "B", "C", "D", "E"]
GRID_COLS = ["W", "X", "Y", "Z"]
ALL_CELLS = [f"{c}{r}" for c in GRID_COLS for r in GRID_ROWS]  # WA…ZE

# Built-in failure modes (seeded from CSV). Users can add more via the form.
DEFAULT_FAILURE_MODES = [
    "welded_to_paper",
    "welded_to_foil",
    "uncut_ends",
    "unknown_endcut_drop",
    "total_line_pickup_fails",
    "wiggly_poor_placement",
]
DEFAULT_FAILURE_LABELS = {
    "welded_to_paper":         "Welded to Paper",
    "welded_to_foil":          "Welded to Foil",
    "uncut_ends":              "Uncut Ends",
    "unknown_endcut_drop":     "Unknown Endcut Drop",
    "total_line_pickup_fails": "Total Line Pickup Fails",
    "wiggly_poor_placement":   "Wiggly/Poor Placement",
}

# Built-in parameter names (always shown in the form)
DEFAULT_PARAMS = [
    "tension", "airflow",
    "encoder_r_real", "tensioner_r_real",
    "encoder_r_plc", "tensioner_r_plc",
]

# ── DB helpers ─────────────────────────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trials (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_date       TEXT,
            trial_num        INTEGER,
            tension          REAL,
            airflow          REAL,
            encoder_r_real   REAL,
            tensioner_r_real REAL,
            encoder_r_plc    REAL,
            tensioner_r_plc  REAL,
            welded_to_paper         TEXT,
            welded_to_foil          TEXT,
            uncut_ends              TEXT,
            unknown_endcut_drop     TEXT,
            total_line_pickup_fails TEXT,
            wiggly_poor_placement   TEXT,
            notes            TEXT
        );

        -- Arbitrary extra numeric parameters per trial (e.g. "voltage", "temperature")
        CREATE TABLE IF NOT EXISTS trial_params (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_id  INTEGER REFERENCES trials(id) ON DELETE CASCADE,
            name      TEXT NOT NULL,
            value     REAL
        );

        -- Normalised per-cell failure records
        CREATE TABLE IF NOT EXISTS failure_cells (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_id     INTEGER REFERENCES trials(id) ON DELETE CASCADE,
            failure_mode TEXT NOT NULL,
            cell         TEXT NOT NULL,
            count        INTEGER DEFAULT 1
        );

        -- Registry of all known failure mode keys (built-in + user-defined)
        CREATE TABLE IF NOT EXISTS failure_mode_registry (
            key   TEXT PRIMARY KEY,
            label TEXT NOT NULL
        );

        -- Registry of all known extra parameter names
        CREATE TABLE IF NOT EXISTS param_registry (
            name TEXT PRIMARY KEY
        );
    """)
    # Add category column to failure_cells if upgrading from old schema
    fc_cols = [r[1] for r in conn.execute("PRAGMA table_info(failure_cells)").fetchall()]
    if "category" not in fc_cols:
        conn.execute("ALTER TABLE failure_cells ADD COLUMN category TEXT")

    # Seed registries with defaults (ignore conflicts)
    for key, label in DEFAULT_FAILURE_LABELS.items():
        conn.execute(
            "INSERT OR IGNORE INTO failure_mode_registry (key, label) VALUES (?,?)",
            (key, label),
        )
    conn.commit()


# ── registry helpers ───────────────────────────────────────────────────────────

def load_failure_registry(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT key, label FROM failure_mode_registry ORDER BY rowid").fetchall()
    return {r["key"]: r["label"] for r in rows}


def load_param_registry(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT name FROM param_registry ORDER BY name").fetchall()
    return [r["name"] for r in rows]


def register_failure_mode(conn: sqlite3.Connection, label: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    conn.execute(
        "INSERT OR IGNORE INTO failure_mode_registry (key, label) VALUES (?,?)",
        (key, label.strip()),
    )
    conn.commit()
    return key


def register_param(conn: sqlite3.Connection, name: str) -> None:
    conn.execute("INSERT OR IGNORE INTO param_registry (name) VALUES (?)", (name.strip(),))
    conn.commit()


# ── cell parsing ───────────────────────────────────────────────────────────────

def parse_failure_cells(raw: str) -> list[tuple[str, int]]:
    """
    Parse free-text like "1WB, 2XC/XD, ZA" into [(cell, count), …].
    Column-only tokens like "XYZ" are skipped (placement notes, not cell refs).
    """
    if not raw or str(raw).strip() in ("0", "", "nan"):
        return []
    results: list[tuple[str, int]] = []
    tokens = re.split(r"[,;\s]+", str(raw).strip())
    for tok in tokens:
        tok = tok.strip()
        if not tok or tok == "0":
            continue
        m = re.match(r"^(\d+)?([A-Za-z].*)$", tok)
        if not m:
            continue
        count_str, body = m.group(1), m.group(2).upper()
        count = int(count_str) if count_str else 1
        cell_tokens = body.split("/")
        cells_found = []
        for ct in cell_tokens:
            ct = ct.strip()
            if re.fullmatch(r"[WXYZ][ABCDE]", ct):
                cells_found.append(ct)
            elif re.fullmatch(r"[ABCDE]", ct) and cells_found:
                cells_found.append(cells_found[-1][0] + ct)
        for cell in cells_found:
            results.append((cell, count))
    return results



def upsert_failure_cells(
    conn: sqlite3.Connection,
    trial_id: int,
    mode_cells: dict[str, list[tuple[str, int, str | None]]],
) -> None:
    """mode_cells: {mode_key: [(cell, count, category), …]}"""
    conn.execute("DELETE FROM failure_cells WHERE trial_id=?", (trial_id,))
    for mode, cell_list in mode_cells.items():
        for cell, cnt, cat in cell_list:
            conn.execute(
                "INSERT INTO failure_cells (trial_id, failure_mode, cell, count, category) VALUES (?,?,?,?,?)",
                (trial_id, mode, cell, cnt, cat),
            )


def upsert_extra_params(
    conn: sqlite3.Connection,
    trial_id: int,
    params: dict[str, float | None],
) -> None:
    conn.execute("DELETE FROM trial_params WHERE trial_id=?", (trial_id,))
    for name, value in params.items():
        if value is not None:
            conn.execute(
                "INSERT INTO trial_params (trial_id, name, value) VALUES (?,?,?)",
                (trial_id, name, value),
            )


# ── CSV seeding ────────────────────────────────────────────────────────────────

def seed_from_csv(conn: sqlite3.Connection) -> None:
    if conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] > 0:
        return
    if not CSV_PATH.exists():
        st.warning(f"CSV not found at {CSV_PATH}. Starting with empty database.")
        return

    df = pd.read_csv(CSV_PATH, header=9, skip_blank_lines=False)
    col_map = {
        df.columns[0]:  "trial_date",
        df.columns[1]:  "trial_num",
        df.columns[2]:  "tension",
        df.columns[3]:  "airflow",
        df.columns[4]:  "encoder_r_real",
        df.columns[5]:  "tensioner_r_real",
        df.columns[6]:  "encoder_r_plc",
        df.columns[7]:  "tensioner_r_plc",
        df.columns[8]:  "welded_to_paper",
        df.columns[9]:  "welded_to_foil",
        df.columns[10]: "uncut_ends",
        df.columns[11]: "unknown_endcut_drop",
        df.columns[12]: "total_line_pickup_fails",
        df.columns[13]: "wiggly_poor_placement",
        df.columns[14]: "notes",
    }
    df = df.rename(columns=col_map)
    df = df[pd.to_numeric(df["trial_num"], errors="coerce").notna()]
    df = df[df["trial_num"] != 56]
    df["trial_num"] = df["trial_num"].astype(int)
    for col in ["tension", "airflow", "encoder_r_real", "tensioner_r_real",
                "encoder_r_plc", "tensioner_r_plc"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for _, r in df.iterrows():
        row_dict = {k: str(r.get(k, "") or "") for k in DEFAULT_FAILURE_MODES}
        row_dict.update({
            "trial_date":      str(r.get("trial_date", "") or ""),
            "trial_num":       int(r["trial_num"]),
            "tension":         r.get("tension"),
            "airflow":         r.get("airflow"),
            "encoder_r_real":  r.get("encoder_r_real"),
            "tensioner_r_real":r.get("tensioner_r_real"),
            "encoder_r_plc":   r.get("encoder_r_plc"),
            "tensioner_r_plc": r.get("tensioner_r_plc"),
            "notes":           str(r.get("notes", "") or ""),
        })
        cur = conn.execute("""
            INSERT INTO trials
                (trial_date, trial_num, tension, airflow,
                 encoder_r_real, tensioner_r_real, encoder_r_plc, tensioner_r_plc,
                 welded_to_paper, welded_to_foil, uncut_ends, unknown_endcut_drop,
                 total_line_pickup_fails, wiggly_poor_placement, notes)
            VALUES
                (:trial_date, :trial_num, :tension, :airflow,
                 :encoder_r_real, :tensioner_r_real, :encoder_r_plc, :tensioner_r_plc,
                 :welded_to_paper, :welded_to_foil, :uncut_ends, :unknown_endcut_drop,
                 :total_line_pickup_fails, :wiggly_poor_placement, :notes)
        """, row_dict)
        tid = cur.lastrowid
        mode_cells = {
            mode: [(cell, cnt, None) for cell, cnt in parse_failure_cells(row_dict[mode])]
            for mode in DEFAULT_FAILURE_MODES
        }
        upsert_failure_cells(conn, tid, mode_cells)
    conn.commit()


# ── data loaders ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_trials() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM trials ORDER BY id", conn)
    conn.close()
    return df


@st.cache_data(ttl=30)
def load_failure_cells() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql("""
        SELECT fc.*, t.tension, t.airflow, t.trial_num, t.trial_date,
               t.encoder_r_real, t.encoder_r_plc,
               t.tensioner_r_real, t.tensioner_r_plc
        FROM failure_cells fc
        JOIN trials t ON fc.trial_id = t.id
    """, conn)
    conn.close()
    return df


@st.cache_data(ttl=30)
def load_extra_params() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM trial_params", conn)
    conn.close()
    return df


def invalidate_cache() -> None:
    load_trials.clear()
    load_failure_cells.clear()
    load_extra_params.clear()


# ── cell-first picker ─────────────────────────────────────────────────────────
# Session-state keys:
#   cf_selected_cell              — which cell is currently open ("WB" or None)
#   cf_{cell}_{mode_key}          — count (int) for a given cell × mode
#   cf_cat_{cell}_{mode_key}      — category str for a given cell × mode entry

def _cf_key(cell: str, mode_key: str) -> str:
    return f"cf_{cell}_{mode_key}"

def _cf_cat_key(cell: str, mode_key: str) -> str:
    return f"cf_cat_{cell}_{mode_key}"

def _cell_total(cell: str, failure_registry: dict) -> int:
    return sum(
        st.session_state.get(_cf_key(cell, mk), 0)
        for mk in failure_registry
    )


def cell_first_picker(failure_registry: dict) -> None:
    """
    Step 1 — 4×5 grid: click a cell to open its failure count panel.
    Step 2 — Categorize: for every non-zero entry assign point/line/multicell.
    """
    selected = st.session_state.get("cf_selected_cell")

    # ── grid ──────────────────────────────────────────────────────────────────
    header_cols = st.columns([1] + [3] * 4)
    for i, col_letter in enumerate(GRID_COLS):
        header_cols[i + 1].markdown(
            f"<div style='text-align:center;font-weight:bold;font-size:1.1em'>"
            f"{col_letter}</div>",
            unsafe_allow_html=True,
        )

    for row in GRID_ROWS:
        cols = st.columns([1] + [3] * 4)
        cols[0].markdown(
            f"<div style='text-align:center;font-weight:bold;padding-top:6px;"
            f"font-size:1.1em'>{row}</div>",
            unsafe_allow_html=True,
        )
        for i, col_letter in enumerate(GRID_COLS):
            cell  = f"{col_letter}{row}"
            total = _cell_total(cell, failure_registry)
            is_open = (cell == selected)

            if total > 0:
                label_btn = f"● {total}"
            elif is_open:
                label_btn = f"[ {cell} ]"
            else:
                label_btn = cell

            if cols[i + 1].button(label_btn, key=f"cfbtn_{cell}",
                                  use_container_width=True):
                st.session_state["cf_selected_cell"] = None if is_open else cell
                st.rerun()

    # ── detail panel ──────────────────────────────────────────────────────────
    if selected:
        st.markdown(f"#### Failures at **{selected}**")
        st.caption("Use − / + to set the count for each failure mode.")

        mode_keys = list(failure_registry.keys())
        for row_start in range(0, len(mode_keys), 3):
            chunk = mode_keys[row_start: row_start + 3]
            detail_cols = st.columns(3)
            for j, mk in enumerate(chunk):
                with detail_cols[j]:
                    skey = _cf_key(selected, mk)
                    cur  = st.session_state.get(skey, 0)
                    st.markdown(
                        f"<div style='font-size:0.82em;font-weight:600;"
                        f"margin-bottom:2px'>{failure_registry[mk]}</div>",
                        unsafe_allow_html=True,
                    )
                    btn_col1, cnt_col, btn_col2 = st.columns([1, 1, 1])
                    if btn_col1.button("−", key=f"cfminus_{selected}_{mk}",
                                      use_container_width=True):
                        st.session_state[skey] = max(0, cur - 1)
                        st.rerun()
                    cnt_col.markdown(
                        f"<div style='text-align:center;font-size:1.3em;"
                        f"padding-top:4px'>{cur}</div>",
                        unsafe_allow_html=True,
                    )
                    if btn_col2.button("+", key=f"cfplus_{selected}_{mk}",
                                      use_container_width=True):
                        st.session_state[skey] = cur + 1
                        st.rerun()

        if st.button("Close panel", key="cf_close"):
            st.session_state["cf_selected_cell"] = None
            st.rerun()

    # ── live summary ──────────────────────────────────────────────────────────
    summary_parts = []
    for mk, mlabel in failure_registry.items():
        entries = []
        for cell in ALL_CELLS:
            cnt = st.session_state.get(_cf_key(cell, mk), 0)
            if cnt > 0:
                entries.append(f"{cnt}{cell}")
        if entries:
            summary_parts.append(f"**{mlabel}:** {', '.join(entries)}")
    if summary_parts:
        st.success("  \n".join(summary_parts))
    else:
        st.caption("No failures recorded yet.")

    # ── Step 2: categorize each non-zero entry ────────────────────────────────
    # Build list of (cell, mode_key, count) that have been filled in
    active_entries = []
    for mk in failure_registry:
        for cell in ALL_CELLS:
            cnt = st.session_state.get(_cf_key(cell, mk), 0)
            if cnt > 0:
                active_entries.append((cell, mk, cnt))

    if active_entries:
        st.markdown("#### Categorize Failures")
        st.caption(
            "For each recorded failure, select whether it is a **point** defect "
            "(single cell), **line** defect (whole column), or **multicell** defect."
        )
        cat_options = ["point", "line", "multicell"]

        for row_start in range(0, len(active_entries), 2):
            chunk = active_entries[row_start: row_start + 2]
            cat_cols = st.columns(2)
            for j, (cell, mk, cnt) in enumerate(chunk):
                ckey = _cf_cat_key(cell, mk)
                cur_cat = st.session_state.get(ckey, "point")
                cur_idx = cat_options.index(cur_cat) if cur_cat in cat_options else 0
                with cat_cols[j]:
                    chosen = st.radio(
                        f"**{failure_registry[mk]}** at {cell} (×{cnt})",
                        options=cat_options,
                        format_func=lambda x: {"point": "Point", "line": "Line", "multicell": "Multicell"}[x],
                        index=cur_idx,
                        key=f"catradio_{cell}_{mk}",
                        horizontal=True,
                    )
                    st.session_state[ckey] = chosen


def clear_cell_first_picker(failure_registry: dict) -> None:
    st.session_state["cf_selected_cell"] = None
    for cell in ALL_CELLS:
        for mk in failure_registry:
            for key in (_cf_key(cell, mk), _cf_cat_key(cell, mk)):
                if key in st.session_state:
                    del st.session_state[key]


def collect_mode_cells(failure_registry: dict) -> dict[str, list[tuple[str, int, str | None]]]:
    """Return {mode_key: [(cell, count, category), …]} from current picker state."""
    result: dict[str, list[tuple[str, int, str | None]]] = {}
    for mk in failure_registry:
        entries = []
        for cell in ALL_CELLS:
            cnt = st.session_state.get(_cf_key(cell, mk), 0)
            if cnt > 0:
                cat = st.session_state.get(_cf_cat_key(cell, mk))
                entries.append((cell, cnt, cat))
        result[mk] = entries
    return result


def picker_summary_for_mode(mode_key: str) -> str:
    """Compact text for storing in the legacy text column, e.g. '1WB, 2XC'."""
    parts = []
    for cell in ALL_CELLS:
        cnt = st.session_state.get(_cf_key(cell, mode_key), 0)
        if cnt > 0:
            parts.append(f"{cnt}{cell}")
    return ", ".join(parts) if parts else "0"


# ── Tab 1: Data Entry ──────────────────────────────────────────────────────────

def _load_last_trial_defaults(conn: sqlite3.Connection) -> dict:
    """Return the most-recently saved trial's parameters as a defaults dict."""
    row = conn.execute(
        "SELECT * FROM trials ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return {}
    d = dict(row)
    # next trial num = last + 1
    d["trial_num"] = (d.get("trial_num") or 0) + 1
    return d


def _default(key: str, fallback, ss_prefix: str = "form_") -> any:
    """Read from session state if present, else return fallback."""
    return st.session_state.get(f"{ss_prefix}{key}", fallback)


def _set_form_defaults(defaults: dict) -> None:
    for k, v in defaults.items():
        st.session_state[f"form_{k}"] = v


def _clear_form_defaults() -> None:
    keys_to_clear = [k for k in st.session_state if k.startswith("form_")]
    for k in keys_to_clear:
        del st.session_state[k]


def tab_data_entry(conn: sqlite3.Connection) -> None:
    st.header("Log a New Trial")

    failure_registry  = load_failure_registry(conn)
    extra_param_names = load_param_registry(conn)

    # Seed form defaults from last trial on first load this session
    if "form_defaults_seeded" not in st.session_state:
        defaults = _load_last_trial_defaults(conn)
        _set_form_defaults(defaults)
        st.session_state["form_defaults_seeded"] = True

    # ── Section 1: core parameters ────────────────────────────────────────────
    with st.form("trial_core_form", clear_on_submit=False):
        st.subheader("Trial Parameters")
        c1, c2 = st.columns(2)
        with c1:
            trial_date = st.date_input("Date", value=date.today())
            trial_num  = st.number_input("Trial #", min_value=1, step=1,
                                         value=int(_default("trial_num", 1)))
            tension    = st.number_input("Tension (N)", min_value=0.0, step=50.0,
                                         value=float(_default("tension", 100.0) or 100.0))
            airflow    = st.number_input("Airflow", min_value=0.0, step=0.5,
                                         value=float(_default("airflow", 5.0) or 5.0))
        with c2:
            encoder_r_real   = st.number_input("Encoder r (real) mm",   format="%.4f",
                                                value=float(_default("encoder_r_real",   0.0) or 0.0))
            tensioner_r_real = st.number_input("Tensioner r (real) mm", format="%.4f",
                                                value=float(_default("tensioner_r_real", 0.0) or 0.0))
            encoder_r_plc    = st.number_input("Encoder r (PLC) mm",    format="%.4f",
                                                value=float(_default("encoder_r_plc",    0.0) or 0.0))
            tensioner_r_plc  = st.number_input("Tensioner r (PLC) mm",  format="%.4f",
                                                value=float(_default("tensioner_r_plc",  0.0) or 0.0))

        # Extra parameters (user-defined)
        extra_values: dict[str, float | None] = {}
        if extra_param_names:
            st.markdown("**Extra Parameters**")
            ep_cols = st.columns(min(len(extra_param_names), 4))
            for i, pname in enumerate(extra_param_names):
                with ep_cols[i % 4]:
                    extra_values[pname] = st.number_input(
                        pname, format="%.4f",
                        value=float(_default(f"ep_{pname}", 0.0) or 0.0),
                        key=f"ep_{pname}",
                    )

        notes = st.text_area("Notes", height=70,
                             value=str(_default("notes", "") or ""))

        save_col, clear_col = st.columns([2, 1])
        with save_col:
            submitted = st.form_submit_button("Save Trial", type="primary",
                                              use_container_width=True)
        with clear_col:
            cleared = st.form_submit_button("Clear All", use_container_width=True)

    # ── Section 2: add a new parameter type ───────────────────────────────────
    with st.expander("➕ Add a new parameter type"):
        new_param = st.text_input("Parameter name (e.g. Voltage, Temperature)",
                                  key="new_param_input")
        if st.button("Add parameter", key="add_param_btn"):
            if new_param.strip():
                register_param(conn, new_param.strip())
                st.success(f"Parameter '{new_param.strip()}' added.")
                st.rerun()

    # ── Section 3: cell-first failure picker ──────────────────────────────────
    st.subheader("Failure Cell Selection")
    st.caption("Click a cell to open its failure detail panel. "
               "Cells with failures show ● and their total count.")
    cell_first_picker(failure_registry)

    # ── Section 4: add a new failure mode ─────────────────────────────────────
    with st.expander("➕ Add a new failure mode"):
        new_mode_label = st.text_input("Failure mode name (e.g. Torn Edge, Misalignment)",
                                       key="new_mode_input")
        if st.button("Add failure mode", key="add_mode_btn"):
            if new_mode_label.strip():
                register_failure_mode(conn, new_mode_label.strip())
                st.success(f"Failure mode '{new_mode_label.strip()}' added.")
                st.rerun()

    # ── Clear logic ────────────────────────────────────────────────────────────
    if cleared:
        _clear_form_defaults()
        clear_cell_first_picker(failure_registry)
        st.rerun()

    # ── Save logic ─────────────────────────────────────────────────────────────
    if submitted:
        trial_date_str = trial_date.strftime("%m/%d").lstrip("0")
        row_dict = {
            "trial_date":      trial_date_str,
            "trial_num":       int(trial_num),
            "tension":         float(tension) or None,
            "airflow":         float(airflow) or None,
            "encoder_r_real":  float(encoder_r_real) or None,
            "tensioner_r_real":float(tensioner_r_real) or None,
            "encoder_r_plc":   float(encoder_r_plc) or None,
            "tensioner_r_plc": float(tensioner_r_plc) or None,
            "notes":           notes,
        }
        # Build raw text for the built-in legacy columns (from picker state)
        for mode_key in DEFAULT_FAILURE_MODES:
            row_dict[mode_key] = picker_summary_for_mode(mode_key)

        cur = conn.execute("""
            INSERT INTO trials
                (trial_date, trial_num, tension, airflow,
                 encoder_r_real, tensioner_r_real, encoder_r_plc, tensioner_r_plc,
                 welded_to_paper, welded_to_foil, uncut_ends, unknown_endcut_drop,
                 total_line_pickup_fails, wiggly_poor_placement, notes)
            VALUES
                (:trial_date, :trial_num, :tension, :airflow,
                 :encoder_r_real, :tensioner_r_real, :encoder_r_plc, :tensioner_r_plc,
                 :welded_to_paper, :welded_to_foil, :uncut_ends, :unknown_endcut_drop,
                 :total_line_pickup_fails, :wiggly_poor_placement, :notes)
        """, row_dict)
        tid = cur.lastrowid

        # Collect all failure cells from the cell-first picker
        mode_cells = collect_mode_cells(failure_registry)
        upsert_failure_cells(conn, tid, mode_cells)

        # Save extra params
        upsert_extra_params(conn, tid, extra_values)

        conn.commit()
        invalidate_cache()
        clear_cell_first_picker(failure_registry)

        # Repopulate form with this trial's params, advancing trial_num by 1
        new_defaults = {
            "trial_num":       int(trial_num) + 1,
            "tension":         row_dict["tension"],
            "airflow":         row_dict["airflow"],
            "encoder_r_real":  row_dict["encoder_r_real"],
            "tensioner_r_real":row_dict["tensioner_r_real"],
            "encoder_r_plc":   row_dict["encoder_r_plc"],
            "tensioner_r_plc": row_dict["tensioner_r_plc"],
            "notes":           "",
        }
        for pname, pval in extra_values.items():
            new_defaults[f"ep_{pname}"] = pval
        _set_form_defaults(new_defaults)

        st.success(f"Trial #{int(trial_num)} saved.")
        st.rerun()

    # ── Recent trials preview ──────────────────────────────────────────────────
    st.subheader("Recent Trials")
    df = load_trials()
    if not df.empty:
        st.dataframe(df.tail(10).iloc[::-1], use_container_width=True)


# ── Tab 2: Dashboard ───────────────────────────────────────────────────────────

def tab_dashboard(conn: sqlite3.Connection) -> None:
    st.header("Dashboard")

    trials  = load_trials()
    fc_df   = load_failure_cells()
    failure_registry = load_failure_registry(conn)
    failure_labels   = failure_registry  # key → label

    if trials.empty:
        st.info("No trial data yet. Add trials in the Data Entry tab.")
        return

    # Parse trial_date strings ("6/17", "6/18" …) into real dates for filtering.
    # Dates have no year in the CSV so we infer year: if month is in the future
    # assume last year, otherwise current year.
    def _parse_trial_date(s: str) -> date | None:
        try:
            d = pd.to_datetime(s, format="%m/%d")
            today = date.today()
            year = today.year if d.month <= today.month else today.year - 1
            return date(year, d.month, d.day)
        except Exception:
            return None

    trials["_date"] = trials["trial_date"].apply(_parse_trial_date)
    valid_dates = trials["_date"].dropna()

    min_date = valid_dates.min() if not valid_dates.empty else date.today()
    max_date = valid_dates.max() if not valid_dates.empty else date.today()

    with st.form("dashboard_filters"):
        with st.expander("Filters", expanded=True):
            row1_c1, row1_c2, row1_c3, row1_c4 = st.columns(4)
            with row1_c1:
                tension_vals = sorted(trials["tension"].dropna().unique())
                sel_tension  = st.multiselect("Tension (N)", tension_vals, default=tension_vals)
            with row1_c2:
                airflow_vals = sorted(trials["airflow"].dropna().unique())
                sel_airflow  = st.multiselect("Airflow", airflow_vals, default=airflow_vals)
            with row1_c3:
                sel_modes = st.multiselect(
                    "Failure Mode (heat map)",
                    list(failure_labels.values()),
                    default=[],
                    placeholder="All modes",
                )
            with row1_c4:
                sel_cells = st.multiselect(
                    "Grid Cell Filter",
                    ALL_CELLS,
                    default=[],
                    placeholder="All cells",
                )

            row2_c1, row2_c2, row2_c3, row2_c4 = st.columns([1, 1, 1, 1])
            with row2_c1:
                all_dates = st.checkbox("All dates", value=True, key="dash_all_dates")
            with row2_c2:
                sel_date_from = st.date_input("Date from", value=min_date,
                                              key="dash_date_from",
                                              disabled=all_dates)
            with row2_c3:
                sel_date_to = st.date_input("Date to", value=max_date,
                                            key="dash_date_to",
                                            disabled=all_dates)

        st.form_submit_button("Apply Filters", type="primary")

    mask = pd.Series(True, index=trials.index)
    if sel_tension:
        mask &= trials["tension"].isin(sel_tension)
    if sel_airflow:
        mask &= trials["airflow"].isin(sel_airflow)
    if not all_dates:
        date_mask = trials["_date"].apply(
            lambda d: (d is None) or (sel_date_from <= d <= sel_date_to)
        )
        mask &= date_mask
    filtered_trials = trials[mask]
    filtered_ids    = set(filtered_trials["id"].tolist())

    # Resolve selected mode keys (empty = all)
    if sel_modes:
        label_to_key = {v: k for k, v in failure_labels.items()}
        sel_mode_keys = [label_to_key[l] for l in sel_modes if l in label_to_key]
    else:
        sel_mode_keys = []

    fc_filtered = fc_df[fc_df["trial_id"].isin(filtered_ids)]
    if sel_mode_keys:
        fc_filtered = fc_filtered[fc_filtered["failure_mode"].isin(sel_mode_keys)]
    if sel_cells:
        fc_filtered = fc_filtered[fc_filtered["cell"].isin(sel_cells)]

    # ── heat map ───────────────────────────────────────────────────────────────
    st.subheader("Grid Heat Map — Failure Count")

    # Per-cell total counts
    cell_totals = (
        fc_filtered.groupby("cell")["count"]
        .sum()
        .reindex(ALL_CELLS, fill_value=0)
    )

    # Per-cell, per-mode breakdown for the hover tooltip
    mode_by_cell = (
        fc_filtered.groupby(["cell", "failure_mode"])["count"]
        .sum()
        .reset_index()
    )
    mode_by_cell["label"] = (
        mode_by_cell["failure_mode"].map(failure_labels)
        .fillna(mode_by_cell["failure_mode"])
    )

    def _hover_text(cell: str) -> str:
        sub = mode_by_cell[mode_by_cell["cell"] == cell].sort_values("count", ascending=False)
        total = int(cell_totals.get(cell, 0))
        if total == 0:
            return f"<b>{cell}</b><br>No failures"
        lines = [f"<b>{cell}</b> — {total} total"]
        for _, r in sub.iterrows():
            lines.append(f"  {r['label']}: {int(r['count'])}")
        return "<br>".join(lines)

    # Build 2-D arrays for z (counts), text (shown on cell), hover
    z_matrix        = []
    text_matrix     = []
    hover_matrix    = []
    for row in GRID_ROWS:
        z_row = []
        t_row = []
        h_row = []
        for col in GRID_COLS:
            cell  = f"{col}{row}"
            total = int(cell_totals.get(cell, 0))
            z_row.append(total)
            t_row.append(str(total) if total > 0 else "")
            h_row.append(_hover_text(cell))
        z_matrix.append(z_row)
        text_matrix.append(t_row)
        hover_matrix.append(h_row)

    fig_hm = go.Figure(go.Heatmap(
        z=z_matrix,
        x=GRID_COLS,
        y=GRID_ROWS,
        text=text_matrix,
        customdata=hover_matrix,
        texttemplate="%{text}",
        hovertemplate="%{customdata}<extra></extra>",
        colorscale="Reds",
        showscale=True,
    ))
    fig_hm.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Column (W→Z)",
        yaxis_title="Row (A→E)",
        yaxis_autorange="reversed",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── bar chart ──────────────────────────────────────────────────────────────
    st.subheader("Failures by Mode")

    region_mode = st.radio(
        "Break down by",
        ["All cells", "Column (W/X/Y/Z)", "Row (A/B/C/D/E)", "Specific cell"],
        key="bar_region_mode",
        horizontal=True,
    )
    if region_mode == "Column (W/X/Y/Z)":
        region_sel = st.multiselect("Select columns", GRID_COLS,
                                    default=GRID_COLS, key="bar_col_sel")
    elif region_mode == "Row (A/B/C/D/E)":
        region_sel = st.multiselect("Select rows", GRID_ROWS,
                                    default=GRID_ROWS, key="bar_row_sel")
    elif region_mode == "Specific cell":
        region_sel = st.multiselect("Select cells", ALL_CELLS,
                                    default=[], key="bar_cell_sel")
    else:
        region_sel = None

    # Start from the already-filtered trials; apply region sub-filter on fc_df
    bar_base = fc_df[fc_df["trial_id"].isin(filtered_ids)].copy()
    bar_base["col"] = bar_base["cell"].str[0]
    bar_base["row"] = bar_base["cell"].str[1]

    if region_mode == "Column (W/X/Y/Z)" and region_sel:
        bar_base = bar_base[bar_base["col"].isin(region_sel)]
    elif region_mode == "Row (A/B/C/D/E)" and region_sel:
        bar_base = bar_base[bar_base["row"].isin(region_sel)]
    elif region_mode == "Specific cell" and region_sel:
        bar_base = bar_base[bar_base["cell"].isin(region_sel)]

    bar_data = bar_base.groupby("failure_mode")["count"].sum().reset_index()
    bar_data["label"] = bar_data["failure_mode"].map(failure_labels).fillna(
        bar_data["failure_mode"]
    )

    if bar_data.empty or bar_data["count"].sum() == 0:
        st.info("No failures for the selected region.")
    else:
        fig_bar = px.bar(
            bar_data.sort_values("count", ascending=False),
            x="label", y="count",
            labels={"label": "Failure Mode", "count": "Total Count"},
            color="count", color_continuous_scale="Reds",
        )
        fig_bar.update_layout(height=350, showlegend=False,
                              margin=dict(l=20, r=20, t=30, b=80))
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── trial log ──────────────────────────────────────────────────────────────
    st.subheader("Trial Log")
    base_cols = ["trial_date", "trial_num", "tension", "airflow",
                 "encoder_r_real", "tensioner_r_real",
                 "welded_to_paper", "welded_to_foil", "uncut_ends",
                 "unknown_endcut_drop", "total_line_pickup_fails",
                 "wiggly_poor_placement", "notes"]
    show_cols = [c for c in base_cols if c in filtered_trials.columns]
    st.dataframe(filtered_trials[show_cols].reset_index(drop=True),
                 use_container_width=True, height=400)


# ── Tab 3: Analysis ────────────────────────────────────────────────────────────

def _total_failures_per_trial(trials: pd.DataFrame, fc_df: pd.DataFrame) -> pd.DataFrame:
    totals = fc_df.groupby("trial_id")["count"].sum().reset_index()
    totals.columns = ["id", "total_failures"]
    return trials.merge(totals, on="id", how="left").fillna({"total_failures": 0})


def tab_analysis(conn: sqlite3.Connection) -> None:
    st.header("Analysis")

    trials = load_trials()
    fc_df  = load_failure_cells()
    ep_df  = load_extra_params()
    failure_registry = load_failure_registry(conn)

    if trials.empty:
        st.info("No data to analyze yet.")
        return

    df = _total_failures_per_trial(trials, fc_df)
    df["encoder_drift"] = (df["encoder_r_plc"] - df["encoder_r_real"]).abs()

    # Pivot extra params wide and join
    if not ep_df.empty:
        ep_wide = ep_df.pivot_table(index="trial_id", columns="name",
                                    values="value", aggfunc="first").reset_index()
        ep_wide = ep_wide.rename(columns={"trial_id": "id"})
        df = df.merge(ep_wide, on="id", how="left")

    # ── 1. Correlation table ───────────────────────────────────────────────────
    st.subheader("Parameter Correlations with Total Failures")

    base_param_cols = {
        "tension":          "Tension (N)",
        "airflow":          "Airflow",
        "encoder_drift":    "Encoder r Drift (|PLC−real|)",
        "encoder_r_real":   "Encoder r (real) mm",
        "tensioner_r_real": "Tensioner r (real) mm",
    }
    extra_names = load_param_registry(conn)
    for n in extra_names:
        base_param_cols[n] = n

    corr_rows = []
    for col, label in base_param_cols.items():
        if col not in df.columns:
            continue
        sub = df[[col, "total_failures"]].dropna()
        if len(sub) < 5:
            continue
        r, p = stats.pearsonr(sub[col], sub["total_failures"])
        corr_rows.append({
            "Parameter": label,
            "Pearson r": round(r, 3),
            "p-value": round(p, 4),
            "Significant (p<0.05)": "Yes" if p < 0.05 else "No",
        })

    if corr_rows:
        st.dataframe(pd.DataFrame(corr_rows).set_index("Parameter"),
                     use_container_width=True)
    else:
        st.info("Not enough numeric data for correlations.")

    # ── 2. Scatter: tension vs failures ───────────────────────────────────────
    st.subheader("Scatter: Tension vs Total Failures")
    sub_t = df.dropna(subset=["tension", "total_failures"])
    if not sub_t.empty:
        fig_sc = px.scatter(
            sub_t, x="tension", y="total_failures",
            color="airflow", hover_data=["trial_num", "trial_date"],
            labels={"tension": "Tension (N)", "total_failures": "Total Failures",
                    "airflow": "Airflow"},
            trendline="ols",
        )
        fig_sc.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── 3. Worst grid cells ────────────────────────────────────────────────────
    st.subheader("Worst Grid Cells (all failure modes)")
    worst = (
        fc_df.groupby("cell")["count"].sum()
        .sort_values(ascending=False).reset_index()
    )
    worst.columns = ["Cell", "Total Failures"]
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.dataframe(worst.set_index("Cell"), use_container_width=True)
    with col_b:
        fig_worst = px.bar(
            worst.head(12), x="Cell", y="Total Failures",
            color="Total Failures", color_continuous_scale="Reds",
        )
        fig_worst.update_layout(height=380, showlegend=False,
                                margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_worst, use_container_width=True)

    # ── 4. Summary stats per tension × airflow ────────────────────────────────
    st.subheader("Summary Stats per (Tension × Airflow)")
    combo = (
        df.groupby(["tension", "airflow"])
        .agg(
            trials=("id", "count"),
            mean_failures=("total_failures", "mean"),
            median_failures=("total_failures", "median"),
            max_failures=("total_failures", "max"),
            failure_rate_pct=("total_failures",
                              lambda x: round(100 * (x > 0).mean(), 1)),
        )
        .reset_index()
        .sort_values("mean_failures", ascending=False)
        .rename(columns={
            "tension": "Tension (N)", "airflow": "Airflow",
            "trials": "# Trials",
            "mean_failures": "Mean Failures",
            "median_failures": "Median Failures",
            "max_failures": "Max Failures",
            "failure_rate_pct": "% Trials with Failure",
        })
    )
    st.dataframe(combo.set_index(["Tension (N)", "Airflow"]), use_container_width=True)

    # ── 5. Failure mode breakdown by tension ──────────────────────────────────
    st.subheader("Failure Counts by Mode and Tension")
    mode_tension = (
        fc_df.groupby(["failure_mode", "tension"])["count"].sum().reset_index()
    )
    mode_tension["mode_label"] = (
        mode_tension["failure_mode"].map(failure_registry)
        .fillna(mode_tension["failure_mode"])
    )
    fig_mt = px.bar(
        mode_tension, x="mode_label", y="count",
        color=mode_tension["tension"].astype(str),
        barmode="group",
        labels={"mode_label": "Failure Mode", "count": "Count", "color": "Tension (N)"},
    )
    fig_mt.update_layout(height=400, legend_title_text="Tension (N)",
                         margin=dict(l=20, r=20, t=30, b=80))
    st.plotly_chart(fig_mt, use_container_width=True)

    # ── 6. Encoder drift over trial sequence ──────────────────────────────────
    st.subheader("Encoder r Drift Over Trial Sequence")
    drift_df = df[["trial_num", "encoder_drift"]].dropna().sort_values("trial_num")
    if not drift_df.empty:
        fig_drift = px.line(
            drift_df, x="trial_num", y="encoder_drift",
            labels={"trial_num": "Trial #", "encoder_drift": "|Encoder PLC − Real| mm"},
            markers=True,
        )
        fig_drift.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_drift, use_container_width=True)
    else:
        st.info("No encoder data with both real and PLC values recorded.")

    # ── 7. Extra parameter vs failures (user-defined params) ──────────────────
    if extra_names:
        st.subheader("Extra Parameters vs Total Failures")
        sel_ep = st.selectbox("Select parameter", extra_names, key="ep_scatter")
        if sel_ep and sel_ep in df.columns:
            sub_ep = df.dropna(subset=[sel_ep, "total_failures"])
            if not sub_ep.empty:
                fig_ep = px.scatter(
                    sub_ep, x=sel_ep, y="total_failures",
                    hover_data=["trial_num"],
                    labels={sel_ep: sel_ep, "total_failures": "Total Failures"},
                    trendline="ols",
                )
                fig_ep.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_ep, use_container_width=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Manufacturing Trial Tracker",
        page_icon="🏭",
        layout="wide",
    )
    st.title("Manufacturing Trial Tracker")

    conn = get_conn()
    init_db(conn)

    if "seeded" not in st.session_state:
        seed_from_csv(conn)
        st.session_state["seeded"] = True

    tab1, tab2, tab3 = st.tabs(["Data Entry", "Dashboard", "Analysis"])

    with tab1:
        tab_data_entry(conn)
    with tab2:
        tab_dashboard(conn)
    with tab3:
        tab_analysis(conn)

    conn.close()


if __name__ == "__main__":
    main()
