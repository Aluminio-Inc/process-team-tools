# Manufacturing Trial Tracker

Streamlit app for logging and analyzing manufacturing trial data, backed by a local SQLite database.

## Setup

**1. Clone / download the repo**, then install dependencies:

```bash
pip install -r requirements.txt
```

**2. Place your seed CSV** at:

```
C:\Users\Tsion\Downloads\Tension and Airflow - Sheet1.csv
```

The app will import it automatically on first launch. If the file isn't present, the app starts with an empty database.

**3. Run the app:**

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`. Keep the terminal open while using it.

---

## Project structure

```
process-team-tools/
├── app.py            # full Streamlit app (entry point)
├── trials.db         # SQLite database (auto-created on first run)
├── requirements.txt
└── README.md
```

---

## Tabs

### Data Entry
Log a new trial. Failure columns accept free-text grid references:

| Format | Meaning |
|---|---|
| `0` | No failures |
| `WB` | 1 failure at cell WB |
| `1WB` | 1 failure at cell WB |
| `2XC` | 2 failures at cell XC |
| `XA, YC` | 1 failure each at XA and YC |
| `2ZB/ZD` | 2 failures each at ZB and ZD |
| `1WB/WD, 2XC` | 1 each at WB and WD, 2 at XC |

The grid is **W/X/Y/Z** (columns) × **A/B/C/D/E** (rows). Column-only annotations like `XYZ` or `WX` are treated as placement notes and are not counted as cell failures.

### Dashboard
- Filter by tension, airflow, failure mode, and grid cell
- Heat map of the W×A grid colored by failure count
- Bar chart of total failures by mode
- Scrollable trial log

### Analysis
- Pearson correlation table: tension, airflow, encoder drift vs. failure rate (with p-values)
- Scatter plot: tension vs. failures, colored by airflow
- Worst grid cells ranked by total failures
- Summary stats (mean, median, max failures; % trials with any failure) per tension × airflow combination
- Failure counts broken down by mode and tension
- Encoder r drift (|PLC − real|) plotted over trial sequence

---

## Database schema

**`trials`** — one row per trial run.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | auto-increment |
| `trial_date` | TEXT | e.g. `6/17` |
| `trial_num` | INTEGER | |
| `tension` | REAL | Newtons |
| `airflow` | REAL | |
| `encoder_r_real` | REAL | mm |
| `tensioner_r_real` | REAL | mm |
| `encoder_r_plc` | REAL | mm |
| `tensioner_r_plc` | REAL | mm |
| `welded_to_paper` … `wiggly_poor_placement` | TEXT | raw failure cell strings |
| `notes` | TEXT | |

**`failure_cells`** — normalized failure records, one row per (trial, mode, cell).

| Column | Type |
|---|---|
| `trial_id` | INTEGER FK → trials.id |
| `failure_mode` | TEXT |
| `cell` | TEXT | e.g. `WB` |
| `count` | INTEGER |
