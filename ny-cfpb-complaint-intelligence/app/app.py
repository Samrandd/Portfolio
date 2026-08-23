"""New York Financial Complaint Signals.

The app uses privacy-safe aggregated CFPB complaint data only.  It never
loads complaint narratives.  Counts and percentage changes are calculated in
Python before any explanation is shown.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MONTHLY_FILE = DATA_DIR / "monthly_complaint_summary.csv"
DAILY_FILE = DATA_DIR / "daily_complaint_summary.csv"

DIMENSIONS = [
    "Product",
    "Issue",
    "Submitted via",
    "Company response to consumer",
    "Timely response?",
]
REQUIRED_COLUMNS = {"Complaints", *DIMENSIONS}
MIN_CURRENT_COMPLAINTS = 20
MIN_PRIOR_COMPLAINTS = 10

NY_BLUE = "#003DA5"
NY_ORANGE = "#F58220"
INK = "#172033"
MUTED = "#637083"
GREEN = "#168A57"
PALE_BLUE = "#F3F7FC"


@st.cache_data(show_spinner=False)
def load_summary(path: str, date_column: str) -> pd.DataFrame:
    """Load and validate one pre-aggregated dashboard file."""
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    frame[date_column] = pd.to_datetime(frame[date_column])
    frame["Complaints"] = pd.to_numeric(frame["Complaints"], errors="raise")
    return frame


def apply_filters(frame: pd.DataFrame, filters: dict[str, list[str]]) -> pd.DataFrame:
    filtered = frame.copy()
    for column, selected in filters.items():
        if selected:
            filtered = filtered[filtered[column].isin(selected)]
    return filtered


def fmt_number(value: float | int) -> str:
    return f"{int(round(value)):,}"


def fmt_change(value: float | int) -> str:
    return f"{value:+,.0f}"


def percentage_change(current: int, previous: int) -> float | None:
    if previous == 0:
        return None
    return (current - previous) / previous * 100


def response_table(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("Company response to consumer", as_index=False)["Complaints"]
        .sum()
        .sort_values("Complaints", ascending=False)
    )


def timely_rate(frame: pd.DataFrame) -> tuple[int, int, float | None]:
    total = int(frame["Complaints"].sum())
    timely = int(frame.loc[frame["Timely response?"].eq("Yes"), "Complaints"].sum())
    return timely, total, (timely / total * 100 if total else None)


def build_evidence_summary(
    current: pd.DataFrame, previous: pd.DataFrame, changes: pd.DataFrame
) -> str:
    """A deterministic explanation, deliberately not an LLM-generated claim."""
    current_total = int(current["Complaints"].sum())
    previous_total = int(previous["Complaints"].sum())
    change = current_total - previous_total
    percent = percentage_change(current_total, previous_total)

    if current_total < MIN_CURRENT_COMPLAINTS or previous_total < MIN_PRIOR_COMPLAINTS:
        return (
            "Insufficient volume for a meaningful 28-day comparison under the "
            "app's minimum-volume guardrail."
        )

    direction = "increased" if change >= 0 else "decreased"
    percent_text = f" ({abs(percent):.1f}%)" if percent is not None else ""
    text = (
        f"Filtered complaint volume {direction} by {abs(change):,}{percent_text}: "
        f"{current_total:,} in the selected 28 days versus {previous_total:,} in the prior 28 days."
    )

    if not changes.empty:
        top = changes.iloc[0]
        if top["Current complaints"] >= MIN_CURRENT_COMPLAINTS:
            text += (
                f" The largest issue-level increase was {top['Issue']} "
                f"({top['Current complaints']:,} vs {top['Previous complaints']:,}; "
                f"{fmt_change(top['Absolute change'])})."
            )
    return text


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background: #FFFFFF; color: {INK}; }}
        [data-testid="stSidebar"] {{ background: {PALE_BLUE}; border-right: 1px solid #DCE5F0; }}
        h1, h2, h3 {{ color: {INK}; letter-spacing: -0.025em; }}
        [data-testid="stMetric"] {{ background: #FFFFFF; border: 1px solid #DCE5F0; border-radius: 10px; padding: 16px; }}
        [data-testid="stMetricLabel"] {{ color: {MUTED}; }}
        .source-note {{ color: {MUTED}; font-size: 0.84rem; }}
        .guardrail {{ background: {PALE_BLUE}; border-left: 4px solid {NY_BLUE}; padding: 14px 16px; border-radius: 4px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def make_changes(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    current_issue = current.groupby("Issue", as_index=False)["Complaints"].sum()
    previous_issue = previous.groupby("Issue", as_index=False)["Complaints"].sum()
    changes = current_issue.merge(previous_issue, on="Issue", how="outer", suffixes=(" current", " previous")).fillna(0)
    changes.columns = ["Issue", "Current complaints", "Previous complaints"]
    changes["Absolute change"] = changes["Current complaints"] - changes["Previous complaints"]
    changes["Percent change"] = changes.apply(
        lambda row: percentage_change(row["Current complaints"], row["Previous complaints"]), axis=1
    )
    return changes.sort_values("Absolute change", ascending=False)


def main() -> None:
    st.set_page_config(page_title="NY Financial Complaint Signals", page_icon="📊", layout="wide")
    inject_css()

    if not MONTHLY_FILE.exists():
        st.error("Missing data/monthly_complaint_summary.csv. Add the exported aggregate file to start the app.")
        st.stop()

    monthly = load_summary(str(MONTHLY_FILE), "Month")
    daily = load_summary(str(DAILY_FILE), "Date received") if DAILY_FILE.exists() else None
    source = daily if daily is not None else monthly
    date_column = "Date received" if daily is not None else "Month"

    st.title("New York Financial Complaint Signals")
    st.caption("Evidence-led monitoring of consumer complaint patterns and company response outcomes.")

    with st.sidebar:
        st.header("Filters")
        st.caption("Product is a filter. The analysis focuses on issue, response outcome, and change over time.")
        filters: dict[str, list[str]] = {}
        for column in DIMENSIONS:
            options = sorted(source[column].dropna().unique().tolist())
            filters[column] = st.multiselect(column, options)

        if daily is not None:
            latest = daily["Date received"].max().date()
            earliest = daily["Date received"].min().date()
            selected_dates = st.date_input(
                "Selected 28-day period",
                value=(latest - pd.Timedelta(days=27), latest),
                min_value=earliest,
                max_value=latest,
            )
            if len(selected_dates) != 2:
                st.warning("Choose both a start and end date.")
                st.stop()
            start_date, end_date = pd.Timestamp(selected_dates[0]), pd.Timestamp(selected_dates[1])
            if (end_date - start_date).days != 27:
                st.warning("Choose exactly 28 days. The preceding 28 days are calculated automatically.")
                st.stop()

    filtered_source = apply_filters(source, filters)
    filtered_monthly = apply_filters(monthly, filters)

    if daily is None:
        total = int(filtered_source["Complaints"].sum())
        timely, response_total, rate = timely_rate(filtered_source)
        st.warning(
            "The monthly trend is ready. Add daily_complaint_summary.csv to activate the exact 28-day comparison and change ranking."
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("Filtered complaints", fmt_number(total))
        col2.metric("Timely responses", fmt_number(timely))
        col3.metric("Timely response rate", f"{rate:.1f}%" if rate is not None else "—")
        insight_frame = filtered_source
    else:
        previous_end = start_date - pd.Timedelta(days=1)
        previous_start = previous_end - pd.Timedelta(days=27)
        current = filtered_source[filtered_source[date_column].between(start_date, end_date)]
        previous = filtered_source[filtered_source[date_column].between(previous_start, previous_end)]
        current_total = int(current["Complaints"].sum())
        previous_total = int(previous["Complaints"].sum())
        change = current_total - previous_total
        percent = percentage_change(current_total, previous_total)
        changes = make_changes(current, previous)

        st.subheader("Selected 28 days vs previous 28 days")
        a, b, c, d = st.columns(4)
        a.metric("Selected complaints", fmt_number(current_total))
        a.caption(f"{start_date:%b %-d}–{end_date:%b %-d, %Y}")
        b.metric("Previous complaints", fmt_number(previous_total))
        b.caption(f"{previous_start:%b %-d}–{previous_end:%b %-d, %Y}")
        c.metric("Absolute change", fmt_change(change))
        d.metric("Percentage change", f"{percent:+.1f}%" if percent is not None else "New / no prior volume")

        st.markdown("### What changed?")
        shown_changes = changes[changes["Current complaints"] >= MIN_CURRENT_COMPLAINTS].head(10).copy()
        if shown_changes.empty:
            st.info("No issue meets the minimum current-volume threshold for a ranked comparison.")
        else:
            chart = px.bar(
                shown_changes.sort_values("Absolute change"),
                x="Absolute change",
                y="Issue",
                orientation="h",
                color="Absolute change",
                color_continuous_scale=[NY_BLUE, "#C9D8F2", NY_ORANGE],
                hover_data={"Current complaints": ":,", "Previous complaints": ":,", "Percent change": ":.1f"},
            )
            chart.update_layout(coloraxis_showscale=False, height=440, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(chart, use_container_width=True)
            table = shown_changes[["Issue", "Current complaints", "Previous complaints", "Absolute change", "Percent change"]].copy()
            table["Percent change"] = table["Percent change"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "New")
            st.dataframe(table, hide_index=True, use_container_width=True)

        st.markdown("### So what?")
        st.markdown(f"<div class='guardrail'>{build_evidence_summary(current, previous, changes)}<br><br><strong>Signal, not proof of causation.</strong> Complaint volume may reflect reporting behavior, seasonality, company actions, or other factors.</div>", unsafe_allow_html=True)
        insight_frame = current

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("### Company response outcomes")
        responses = response_table(insight_frame)
        response_chart = px.bar(
            responses,
            x="Complaints",
            y="Company response to consumer",
            orientation="h",
            color="Company response to consumer",
            color_discrete_sequence=[NY_BLUE, NY_ORANGE, GREEN, "#A9B4C4"],
        )
        response_chart.update_layout(showlegend=False, height=340, yaxis_title=None, xaxis_title="Complaints", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(response_chart, use_container_width=True)
        st.caption("“Monetary relief” counts outcomes; the CFPB file does not contain payment amounts.")

    with right:
        st.markdown("### Timely response")
        timely, total, rate = timely_rate(insight_frame)
        st.metric("Timely response rate", f"{rate:.1f}%" if rate is not None else "—", f"{fmt_number(timely)} of {fmt_number(total)} complaints")
        st.markdown(
            "Timely response is a service metric, not a measure of whether the underlying consumer problem was resolved favorably."
        )

    st.markdown("### Monthly complaint trend")
    monthly_trend = (
        filtered_monthly.groupby("Month", as_index=False)["Complaints"].sum().sort_values("Month")
    )
    line = px.line(monthly_trend, x="Month", y="Complaints", markers=True, color_discrete_sequence=[NY_BLUE])
    line.update_layout(height=360, yaxis_title="Complaints", xaxis_title=None, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(line, use_container_width=True)

    st.markdown(
        "<p class='source-note'>Source: CFPB Consumer Complaint Database. Data shown are aggregated New York complaint records. Complaint counts are signals for investigation, not proof of company performance or causation.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
