"""
Local Streamlit UI for WC match predictions.

Runs entirely on your machine — no data is sent to external services.

Start:
    uv run streamlit run src/ui/app.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from etl import custom_fixtures  # noqa: E402
from etl.paths import PROCESSED_DIR  # noqa: E402

WC_PREDICTIONS = PROCESSED_DIR / "wc2026_predictions.csv"
CUSTOM_PREDICTIONS = PROCESSED_DIR / "custom_predictions.csv"
FEATURE_STORE = PROCESSED_DIR / "matches_features.parquet"
ELO_HISTORY = PROCESSED_DIR / "elo_history.parquet"


def _run_script(relative_path: str, *args: str) -> tuple[bool, str]:
    """Run a project script with uv; return (ok, combined output)."""
    cmd = ["uv", "run", "python", relative_path, *args]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output.strip()


@st.cache_data(ttl=30)
def _load_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["date"])


@st.cache_data(ttl=30)
def _load_feature_store() -> pd.DataFrame | None:
    if not FEATURE_STORE.exists():
        return None
    return pd.read_parquet(FEATURE_STORE)


@st.cache_data(ttl=30)
def _load_elo_history() -> pd.DataFrame | None:
    if not ELO_HISTORY.exists():
        return None
    return pd.read_parquet(ELO_HISTORY)


def _dataset_ready() -> bool:
    return FEATURE_STORE.exists()


def _prob_chart(df: pd.DataFrame, title: str) -> None:
    long = df.melt(
        id_vars=["home_team", "away_team", "date", "predicted"],
        value_vars=["p_H", "p_D", "p_A"],
        var_name="outcome",
        value_name="probability",
    )
    long["match"] = long["home_team"] + " vs " + long["away_team"]
    long["outcome"] = long["outcome"].str.replace("p_", "")
    fig = px.bar(
        long,
        x="match",
        y="probability",
        color="outcome",
        barmode="group",
        title=title,
        labels={"probability": "Probability", "match": "Match"},
    )
    fig.update_layout(xaxis_tickangle=-45, height=420)
    st.plotly_chart(fig, use_container_width=True)


def _team_snapshot(team: str, features: pd.DataFrame, elo: pd.DataFrame) -> dict:
    """Latest pre-match state for a team from played matches."""
    played = features[features["played"]].copy()
    home_rows = played[played["home_team"] == team]
    away_rows = played[played["away_team"] == team]

    snapshots = []
    if not home_rows.empty:
        last_h = home_rows.sort_values("date").iloc[-1]
        snapshots.append({
            "date": last_h["date"],
            "elo": last_h["elo_home_pre"],
            "form_pts": last_h["form_pts_home"],
            "fifa_rank": last_h.get("home_fifa_rank"),
        })
    if not away_rows.empty:
        last_a = away_rows.sort_values("date").iloc[-1]
        snapshots.append({
            "date": last_a["date"],
            "elo": last_a["elo_away_pre"],
            "form_pts": last_a["form_pts_away"],
            "fifa_rank": last_a.get("away_fifa_rank"),
        })

    latest_elo = None
    if elo is not None:
        team_elo = elo[elo["team"] == team].sort_values("date")
        if not team_elo.empty:
            latest_elo = float(team_elo.iloc[-1]["elo_pre"])

    if not snapshots:
        return {"elo": latest_elo}

    best = max(snapshots, key=lambda s: s["date"])
    if latest_elo is not None:
        best["elo"] = latest_elo
    return best


def page_wc_predictions() -> None:
    st.subheader("World Cup 2026 — group stage")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh WC predictions", type="primary"):
            with st.spinner("Training model and scoring fixtures..."):
                ok, out = _run_script("src/models/predict_fixtures.py")
            if ok:
                st.success("Predictions updated.")
                _load_csv.clear()
            else:
                st.error("Prediction failed.")
            if out:
                st.code(out[-2000:])

    df = _load_csv(str(WC_PREDICTIONS))
    if df is None:
        st.info(
            "No predictions yet. Run the data pipeline first, then click "
            "**Refresh WC predictions**."
        )
        if st.button("Build dataset (first-time setup)"):
            with st.spinner("Downloading data and building features (~1-2 min)..."):
                ok, out = _run_script(
                    "src/etl/make_dataset.py", "--skip-download", "--skip-scrape"
                )
            st.code(out[-2000:])
            if ok:
                st.success("Dataset ready. Now click Refresh WC predictions.")
                _load_feature_store.clear()
            else:
                st.error("Pipeline failed. Try without --skip-download if raw data is missing.")
        return

    st.caption(f"{len(df)} fixtures — source: `{WC_PREDICTIONS.name}`")
    st.dataframe(
        df[["date", "home_team", "away_team", "predicted", "p_H", "p_D", "p_A", "confidence"]],
        use_container_width=True,
        hide_index=True,
    )
    _prob_chart(df, "WC 2026 outcome probabilities")


def page_custom_fixtures() -> None:
    st.subheader("Custom hypothetical matches")

    features = _load_feature_store()
    if features is None:
        st.warning("Feature store missing. Build the dataset from the WC tab first.")
        return

    teams = sorted(
        set(features["home_team"].dropna()) | set(features["away_team"].dropna())
    )

    with st.form("add_fixture"):
        c1, c2 = st.columns(2)
        home = c1.selectbox("Home team", teams, index=teams.index("Spain") if "Spain" in teams else 0)
        away = c2.selectbox("Away team", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
        c3, c4 = st.columns(2)
        match_date = c3.date_input("Match date", value=pd.Timestamp("2026-07-15").date())
        neutral = c4.checkbox("Neutral venue", value=True)
        submitted = st.form_submit_button("Add & predict", type="primary")

    if submitted:
        if home == away:
            st.error("Home and away must be different.")
        else:
            date_arg = match_date.isoformat()
            with st.spinner("Adding fixture and scoring..."):
                ok, out = _run_script(
                    "src/etl/add_fixture.py",
                    "add",
                    "--home", home,
                    "--away", away,
                    "--date", date_arg,
                    *(["--home-advantage"] if not neutral else []),
                    "--predict",
                )
            _load_csv.clear()
            _load_feature_store.clear()
            if ok:
                st.success(f"Scored: {home} vs {away}")
            else:
                st.error("Failed to add or predict.")
            if out:
                st.code(out[-2000:])

    stored = custom_fixtures.list_fixtures()
    st.markdown("#### Saved custom fixtures")
    if stored.empty:
        st.caption("None yet.")
    else:
        show = stored.copy()
        show["date"] = pd.to_datetime(show["date"]).dt.date
        st.dataframe(show, use_container_width=True, hide_index=True)

        remove_id = st.selectbox("Remove fixture", stored["fixture_id"].tolist())
        if st.button("Remove selected"):
            ok, out = _run_script(
                "src/etl/add_fixture.py", "remove", "--id", remove_id, "--rebuild"
            )
            if stored.shape[0] > 1:
                _run_script("src/etl/add_fixture.py", "predict")
            _load_csv.clear()
            _load_feature_store.clear()
            if ok:
                st.rerun()
            else:
                st.error("Remove failed.")
                st.code(out[-2000:])

    df = _load_csv(str(CUSTOM_PREDICTIONS))
    if df is not None and not df.empty:
        st.markdown("#### Latest predictions")
        st.dataframe(
            df[["date", "home_team", "away_team", "predicted", "p_H", "p_D", "p_A", "confidence"]],
            use_container_width=True,
            hide_index=True,
        )
        if len(df) <= 12:
            _prob_chart(df, "Custom fixture probabilities")


def page_teams() -> None:
    st.subheader("Team strength snapshot")
    features = _load_feature_store()
    elo = _load_elo_history()
    if features is None:
        st.warning("Feature store missing. Build the dataset first.")
        return

    teams = sorted(
        set(features["home_team"].dropna()) | set(features["away_team"].dropna())
    )
    team = st.selectbox("Team", teams, index=teams.index("Spain") if "Spain" in teams else 0)
    snap = _team_snapshot(team, features, elo)

    c1, c2, c3 = st.columns(3)
    c1.metric("Elo rating", f"{snap.get('elo', 0):.0f}" if snap.get("elo") else "—")
    c2.metric("Form (pts)", f"{snap.get('form_pts', 0):.2f}" if snap.get("form_pts") else "—")
    rank = snap.get("fifa_rank")
    c3.metric("FIFA rank", f"{int(rank)}" if pd.notna(rank) else "—")
    if snap.get("date") is not None:
        st.caption(f"Based on last appearance: {pd.Timestamp(snap['date']).date()}")

    if elo is not None:
        hist = elo[elo["team"] == team].sort_values("date")
        if not hist.empty:
            fig = px.line(
                hist, x="date", y="elo_pre", title=f"{team} — Elo over time",
                labels={"elo_pre": "Elo", "date": "Date"},
            )
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="WC 2026 Predictions",
        page_icon="⚽",
        layout="wide",
    )
    st.title("FIFA World Cup Match Predictor")
    st.caption("Local-only · no data sent to external services")

    with st.sidebar:
        st.header("Status")
        ready = _dataset_ready()
        st.write("Dataset", "✅ ready" if ready else "❌ missing")
        st.write("WC predictions", "✅" if WC_PREDICTIONS.exists() else "—")
        st.write("Custom predictions", "✅" if CUSTOM_PREDICTIONS.exists() else "—")
        st.divider()
        st.markdown(
            "**Offline tip:** after the first `make_dataset.py`, everything runs "
            "without internet."
        )

    tab_wc, tab_custom, tab_teams = st.tabs([
        "WC 2026 fixtures",
        "Custom matches",
        "Team explorer",
    ])
    with tab_wc:
        page_wc_predictions()
    with tab_custom:
        page_custom_fixtures()
    with tab_teams:
        page_teams()


if __name__ == "__main__":
    main()
