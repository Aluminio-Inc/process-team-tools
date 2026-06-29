"""
Manufacturing Trial Tracker — Streamlit + SQLite app.

Tabs:
  1. Data Entry  — log a new trial
  2. Dashboard   — heat map, bar chart, trial log with filters
  3. Analysis    — correlations, worst grid cells, summary stats

DB is seeded from CSV on first run if the trials table is empty.
"""

import re
import sqlite3
import os
from pathlib import Path
from datetime import date, datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DB_PATH  = BASE_DIR / "trials.db"
CSV_PATH = Path(r"C:\Users\Tsion\Downloads\Tension and Airflow - Sheet1.csv")

# ── constants ─────────────────────────────────────────────────────────────────
GRID_ROWS   = ["A", "B", "C", "D", "E"]
GRID_COLS   = ["W", "X", "Y", "Z"]
ALL_CELLS   = [f"{c}{r}" for c in GRID_COLS for r in GRID_ROWS]   # WA…ZE

FAILURE_MODES = [
    "welded_to_paper",
    "welded_to_foil",
    "uncut_ends",
    "unknown_endcut_drop",
    "total_line_pickup_fails",
    "wiggly_poor_placement",
]
FAILURE_LABELS = {
    "welded_to_paper":          "Welded to Paper",
    "welded_to_foil":           "Welded to Foil",
    "uncut_ends":               "Uncut Ends",
    "unknown_endcut_drop":      "Unknown Endcut Drop",
    "total_line_pickup_fails":  "Total Line Pickup Fails",
    "wiggly_poor_placement":    "Wiggly/Poor Placement",
}

# ── DB helpers ────────────────────────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trials (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_date              TEXT,
            trial_num               INTEGER,
            tension                 REAL,
            airflow                 REAL,
            encoder_r_real          REAL,
            tensioner_r_real        REAL,
            encoder_r_plc           REAL,
            tensioner_r_plc         REAL,
            welded_to_paper         TEXT,
            welded_to_foil          TEXT,
            uncut_ends              TEXT,
            unknown_endcut_drop     TEXT,
            total_line_pickup_fails TEXT,
            wiggly_poor_placement   TEXT,
            notes                   TEXT
        );

        CREATE TABLE IF NOT EXISTS failure_cells (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_id     INTEGER REFERENCES trials(id) ON DELETE CASCADE,
            failure_mode TEXT,
            cell         TEXT,
            count        INTEGER DEFAULT 1
        );
    """)
    conn.commit()


# ── cell parsing ──────────────────────────────────────────────────────────────

def _expand_slash(token: str) -> list[str]:
    """'XB/XD' → ['XB','XD'];  'WB/WD' works too."""
    parts = token.split("/")
    expanded = []
    for p in parts:
        p = p.strip().upper()
        if re.fullmatch(r"[WXYZ][ABCDE]", p):
            expanded.append(p)
        elif re.fullmatch(r"[WXYZ]", p) and expanded:
            # bare column like the second part of "WB/WD" already handled
            pass
    return expanded if expanded else []


def parse_failure_cells(raw: str) -> list[tuple[str, int]]:
    """
    Parse free-text like "1WB, 2XC/XD, ZA" into [(cell, count), …].
    Returns a flat list; cells in a slash-group each get the leading count.
    Rows/columns without specific cells (e.g. "XYZ", "WX") are skipped —
    those represent wiggly placement annotations, not grid failure cells.
    """
    if not raw or str(raw).strip() in ("0", "", "nan"):
        return []

    results: list[tuple[str, int]] = []

    # split on commas and whitespace runs
    tokens = re.split(r"[,;\s]+", str(raw).strip())

    for tok in tokens:
        tok = tok.strip()
        if not tok or tok == "0":
            continue

        # optional leading count digit
        m = re.match(r"^(\d+)?([A-Za-z].*)$", tok)
        if not m:
            continue
        count_str, body = m.group(1), m.group(2).upper()

        count = int(count_str) if count_str else 1

        # expand slash-separated cells
        cell_tokens = body.split("/")
        cells_found = []
        for ct in cell_tokens:
            ct = ct.strip()
            if re.fullmatch(r"[WXYZ][ABCDE]", ct):
                cells_found.append(ct)
            elif re.fullmatch(r"[ABCDE]", ct) and cells_found:
                # bare row appended to last column, e.g. "WB/D" → WD
                cells_found.append(cells_found[-1][0] + ct)
            # single-letter col-only or row-only tokens are skipped
        for cell in cells_found:
            results.append((cell, count))

    return results


def upsert_failure_cells(conn: sqlite3.Connection, trial_id: int, row: dict) -> None:
    conn.execute("DELETE FROM failure_cells WHERE trial_id=?", (trial_id,))
    for mode in FAILURE_MODES:
        raw = str(row.get(mode) or "")
        for cell, cnt in parse_failure_cells(raw):
            conn.execute(
                "INSERT INTO failure_cells (trial_id, failure_mode, cell, count) VALUES (?,?,?,?)",
                (trial_id, mode, cell, cnt),
            )


# ── CSV seeding ───────────────────────────────────────────────────────────────

def seed_from_csv(conn: sqlite3.Connection) -> None:
    existing = conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
    if existing > 0:
        return

    if not CSV_PATH.exists():
        st.warning(f"CSV not found at {CSV_PATH}. Starting with empty database.")
        return

    # Row 10 (0-indexed row 9) has the headers; rows 0-8 are the legend section
    df = pd.read_csv(CSV_PATH, header=9, skip_blank_lines=False)

    # Rename columns to DB-friendly names
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

    # Drop the trailing blank / summary rows
    df = df[df["trial_num"].notna() & (df["trial_num"] != 56)]
    df = df[pd.to_numeric(df["trial_num"], errors="coerce").notna()]
    df["trial_num"] = df["trial_num"].astype(int)

    numeric_cols = ["tension", "airflow", "encoder_r_real", "tensioner_r_real",
                    "encoder_r_plc", "tensioner_r_plc"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for _, r in df.iterrows():
        row_dict = {
            "trial_date":              str(r.get("trial_date", "") or ""),
            "trial_num":               int(r["trial_num"]),
            "tension":                 r.get("tension"),
            "airflow":                 r.get("airflow"),
            "encoder_r_real":          r.get("encoder_r_real"),
            "tensioner_r_real":        r.get("tensioner_r_real"),
            "encoder_r_plc":           r.get("encoder_r_plc"),
            "tensioner_r_plc":         r.get("tensioner_r_plc"),
            "welded_to_paper":         str(r.get("welded_to_paper", "") or ""),
            "welded_to_foil":          str(r.get("welded_to_foil", "") or ""),
            "uncut_ends":              str(r.get("uncut_ends", "") or ""),
            "unknown_endcut_drop":     str(r.get("unknown_endcut_drop", "") or ""),
            "total_line_pickup_fails": str(r.get("total_line_pickup_fails", "") or ""),
            "wiggly_poor_placement":   str(r.get("wiggly_poor_placement", "") or ""),
            "notes":                   str(r.get("notes", "") or ""),
        }
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
        upsert_failure_cells(conn, cur.lastrowid, row_dict)
    conn.commit()


# ── data loaders ──────────────────────────────────────────────────────────────

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


def invalidate_cache():
    load_trials.clear()
    load_failure_cells.clear()


# ── Tab 1: Data Entry ─────────────────────────────────────────────────────────

def tab_data_entry(conn: sqlite3.Connection) -> None:
    st.header("Log a New Trial")

    with st.form("trial_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            trial_date = st.date_input("Date", value=date.today())
            trial_num  = st.number_input("Trial #", min_value=1, step=1, value=1)
        with c2:
            tension  = st.number_input("Tension (N)", min_value=0.0, step=50.0, value=100.0)
            airflow  = st.number_input("Airflow",     min_value=0.0, step=0.5,  value=5.0)
        with c3:
            encoder_r_real    = st.number_input("Encoder r (real) mm",    value=0.0, format="%.4f")
            tensioner_r_real  = st.number_input("Tensioner r (real) mm",  value=0.0, format="%.4f")
            encoder_r_plc     = st.number_input("Encoder r (PLC) mm",     value=0.0, format="%.4f")
            tensioner_r_plc   = st.number_input("Tensioner r (PLC) mm",   value=0.0, format="%.4f")

        st.markdown("**Failure cells** — enter grid refs like `1WB`, `XA, YC`, `2ZB/ZD`, or `0` for none")
        fc1, fc2 = st.columns(2)
        with fc1:
            welded_to_paper        = st.text_input("Welded to Paper",         value="0")
            welded_to_foil         = st.text_input("Welded to Foil",          value="0")
            uncut_ends             = st.text_input("Uncut Ends",               value="0")
        with fc2:
            unknown_endcut_drop    = st.text_input("Unknown Endcut Drop",      value="0")
            total_line_pickup_fails= st.text_input("Total Line Pickup Fails",  value="0")
            wiggly_poor_placement  = st.text_input("Wiggly/Poor Placement",    value="0")

        notes = st.text_area("Notes", height=80)

        submitted = st.form_submit_button("Save Trial", type="primary")

    if submitted:
        row_dict = {
            "trial_date":              trial_date.strftime("%-m/%d") if os.name != "nt"
                                       else trial_date.strftime("%m/%d").lstrip("0"),
            "trial_num":               int(trial_num),
            "tension":                 float(tension) if tension else None,
            "airflow":                 float(airflow) if airflow else None,
            "encoder_r_real":          float(encoder_r_real) or None,
            "tensioner_r_real":        float(tensioner_r_real) or None,
            "encoder_r_plc":           float(encoder_r_plc) or None,
            "tensioner_r_plc":         float(tensioner_r_plc) or None,
            "welded_to_paper":         welded_to_paper,
            "welded_to_foil":          welded_to_foil,
            "uncut_ends":              uncut_ends,
            "unknown_endcut_drop":     unknown_endcut_drop,
            "total_line_pickup_fails": total_line_pickup_fails,
            "wiggly_poor_placement":   wiggly_poor_placement,
            "notes":                   notes,
        }
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
        upsert_failure_cells(conn, cur.lastrowid, row_dict)
        conn.commit()
        invalidate_cache()
        st.success(f"Trial #{int(trial_num)} saved.")

    # preview recent entries
    st.subheader("Recent Trials")
    df = load_trials()
    if not df.empty:
        st.dataframe(df.tail(10).iloc[::-1], use_container_width=True)


# ── Tab 2: Dashboard ──────────────────────────────────────────────────────────

def tab_dashboard() -> None:
    st.header("Dashboard")

    trials = load_trials()
    fc_df  = load_failure_cells()

    if trials.empty:
        st.info("No trial data yet. Add trials in the Data Entry tab.")
        return

    # ── sidebar-style filters inside the tab ──────────────────────────────────
    with st.expander("Filters", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            tension_vals = sorted(trials["tension"].dropna().unique())
            sel_tension  = st.multiselect("Tension (N)", tension_vals, default=tension_vals)
        with f2:
            airflow_vals = sorted(trials["airflow"].dropna().unique())
            sel_airflow  = st.multiselect("Airflow", airflow_vals, default=airflow_vals)
        with f3:
            sel_mode = st.selectbox("Failure Mode (heat map)",
                                    ["All"] + [FAILURE_LABELS[m] for m in FAILURE_MODES])
        with f4:
            sel_cell = st.selectbox("Grid Cell Filter", ["All"] + ALL_CELLS)

    # filter trials
    mask = pd.Series(True, index=trials.index)
    if sel_tension:
        mask &= trials["tension"].isin(sel_tension)
    if sel_airflow:
        mask &= trials["airflow"].isin(sel_airflow)
    filtered_trials = trials[mask]
    filtered_ids    = set(filtered_trials["id"].tolist())

    # filter failure cells
    fc_filtered = fc_df[fc_df["trial_id"].isin(filtered_ids)]
    if sel_mode != "All":
        mode_key = {v: k for k, v in FAILURE_LABELS.items()}[sel_mode]
        fc_filtered = fc_filtered[fc_filtered["failure_mode"] == mode_key]
    if sel_cell != "All":
        fc_filtered = fc_filtered[fc_filtered["cell"] == sel_cell]

    # ── heat map ──────────────────────────────────────────────────────────────
    st.subheader("Grid Heat Map — Failure Count")
    hm_data = (
        fc_filtered.groupby("cell")["count"]
        .sum()
        .reindex(ALL_CELLS, fill_value=0)
        .reset_index()
    )
    hm_data.columns = ["cell", "failures"]
    hm_data["col"] = hm_data["cell"].str[0]
    hm_data["row"] = hm_data["cell"].str[1]

    pivot = hm_data.pivot(index="row", columns="col", values="failures").reindex(
        index=GRID_ROWS, columns=GRID_COLS, fill_value=0
    )

    fig_hm = px.imshow(
        pivot,
        color_continuous_scale="Reds",
        labels=dict(x="Column (W→Z)", y="Row (A→E)", color="Failures"),
        text_auto=True,
        aspect="equal",
    )
    fig_hm.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── bar chart ─────────────────────────────────────────────────────────────
    st.subheader("Failures by Mode")
    bar_data = (
        fc_df[fc_df["trial_id"].isin(filtered_ids)]
        .groupby("failure_mode")["count"]
        .sum()
        .reset_index()
    )
    bar_data["label"] = bar_data["failure_mode"].map(FAILURE_LABELS)
    fig_bar = px.bar(
        bar_data.sort_values("count", ascending=False),
        x="label", y="count",
        labels={"label": "Failure Mode", "count": "Total Count"},
        color="count", color_continuous_scale="Reds",
    )
    fig_bar.update_layout(height=350, showlegend=False,
                          margin=dict(l=20, r=20, t=30, b=80))
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── trial log ─────────────────────────────────────────────────────────────
    st.subheader("Trial Log")
    display_cols = ["trial_date", "trial_num", "tension", "airflow",
                    "encoder_r_real", "tensioner_r_real",
                    "welded_to_paper", "welded_to_foil", "uncut_ends",
                    "unknown_endcut_drop", "total_line_pickup_fails",
                    "wiggly_poor_placement", "notes"]
    st.dataframe(
        filtered_trials[display_cols].reset_index(drop=True),
        use_container_width=True, height=400,
    )


# ── Tab 3: Analysis ───────────────────────────────────────────────────────────

def _total_failures_per_trial(trials: pd.DataFrame, fc_df: pd.DataFrame) -> pd.DataFrame:
    """Attach a numeric total_failures column to each trial."""
    totals = fc_df.groupby("trial_id")["count"].sum().reset_index()
    totals.columns = ["id", "total_failures"]
    return trials.merge(totals, on="id", how="left").fillna({"total_failures": 0})


def tab_analysis() -> None:
    st.header("Analysis")

    trials = load_trials()
    fc_df  = load_failure_cells()

    if trials.empty:
        st.info("No data to analyze yet.")
        return

    df = _total_failures_per_trial(trials, fc_df)
    df["encoder_drift"] = (df["encoder_r_plc"] - df["encoder_r_real"]).abs()

    # ── 1. Correlation table ──────────────────────────────────────────────────
    st.subheader("Parameter Correlations with Total Failures")

    param_cols = {
        "tension":       "Tension (N)",
        "airflow":       "Airflow",
        "encoder_drift": "Encoder r Drift (|PLC−real|)",
        "encoder_r_real":"Encoder r (real) mm",
        "tensioner_r_real": "Tensioner r (real) mm",
    }

    corr_rows = []
    for col, label in param_cols.items():
        sub = df[[col, "total_failures"]].dropna()
        if len(sub) < 5:
            continue
        r, p = stats.pearsonr(sub[col], sub["total_failures"])
        corr_rows.append({"Parameter": label, "Pearson r": round(r, 3),
                          "p-value": round(p, 4),
                          "Significant (p<0.05)": "Yes" if p < 0.05 else "No"})

    if corr_rows:
        st.dataframe(pd.DataFrame(corr_rows).set_index("Parameter"),
                     use_container_width=True)
    else:
        st.info("Not enough numeric data for correlations.")

    # ── 2. Scatter: tension vs failures ──────────────────────────────────────
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

    # ── 3. Worst grid cells ───────────────────────────────────────────────────
    st.subheader("Worst Grid Cells (all failure modes)")
    worst = (
        fc_df.groupby("cell")["count"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
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

    # ── 4. Summary stats per parameter combination ────────────────────────────
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

    # ── 5. Failure mode breakdown by tension ─────────────────────────────────
    st.subheader("Failure Counts by Mode and Tension")
    mode_tension = (
        fc_df
        .groupby(["failure_mode", "tension"])["count"]
        .sum()
        .reset_index()
    )
    mode_tension["mode_label"] = mode_tension["failure_mode"].map(FAILURE_LABELS)
    fig_mt = px.bar(
        mode_tension, x="mode_label", y="count",
        color=mode_tension["tension"].astype(str),
        barmode="group",
        labels={"mode_label": "Failure Mode", "count": "Count",
                "color": "Tension (N)"},
    )
    fig_mt.update_layout(height=400, legend_title_text="Tension (N)",
                         margin=dict(l=20, r=20, t=30, b=80))
    st.plotly_chart(fig_mt, use_container_width=True)

    # ── 6. Encoder drift over trial sequence ─────────────────────────────────
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
        tab_dashboard()
    with tab3:
        tab_analysis()

    conn.close()


if __name__ == "__main__":
    main()
