"""Cost and quality dashboard over the request log.

Usage: streamlit run dashboard/app.py
Reads data/requests.db (written by every API request and eval run). Pick runs to compare at the top.
"""
import sqlite3
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag import config  # noqa: E402

# Validated categorical palette (fixed order, one colour per way a request was served).
SERVED_ORDER = ["Cache", "Small model", "Large model", "Escalated"]
SERVED_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

st.set_page_config(page_title="RAG cost & quality", layout="wide")


@st.cache_data(ttl=30)
def load() -> pd.DataFrame:
    if not config.REQUEST_DB.exists():
        return pd.DataFrame()
    with sqlite3.connect(config.REQUEST_DB) as conn:
        df = pd.read_sql_query("SELECT * FROM requests ORDER BY ts", conn)
    df["run"] = df["run_id"].fillna("live traffic")
    df["served_by"] = "Large model"
    df.loc[df["model"] == config.SMALL_MODEL, "served_by"] = "Small model"
    df.loc[df["escalated"] == 1, "served_by"] = "Escalated"
    df.loc[df["cache_hit"] == 1, "served_by"] = "Cache"
    df["n"] = df.groupby("run").cumcount() + 1
    df["cumulative_cost"] = df.groupby("run")["cost_usd"].cumsum()
    return df


df = load()
st.title("RAG cost & quality")
if df.empty:
    st.info(f"No requests logged yet in {config.REQUEST_DB}. Run the API or an eval.")
    st.stop()

runs = list(dict.fromkeys(df["run"]))
default = [r for r in runs if "p1-" in r][-2:] or runs[-2:]
selected = st.multiselect("Runs", runs, default=default)
view = df[df["run"].isin(selected)]
if view.empty:
    st.stop()
run_scale = alt.Scale(domain=selected, range=RUN_COLORS[:len(selected)])
served_scale = alt.Scale(domain=SERVED_ORDER, range=SERVED_COLORS)


def summary(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "requests": len(g),
        "cost / request ($)": g["cost_usd"].mean(),
        "total cost ($)": g["cost_usd"].sum(),
        "cache hit rate": g["cache_hit"].mean(),
        "p50 latency (ms)": g["total_ms"].quantile(0.5),
        "p95 latency (ms)": g["total_ms"].quantile(0.95),
        "accuracy": g["correct"].mean() if g["correct"].notna().any() else float("nan"),
    })


table = view.groupby("run", sort=False).apply(summary, include_groups=False)

# Headline tiles, one row per run
for run, row in table.iterrows():
    st.caption(run)
    cols = st.columns(6)
    cols[0].metric("Requests", f"{row['requests']:.0f}")
    cols[1].metric("Cost / request", f"${row['cost / request ($)']:.4f}")
    cols[2].metric("Cache hit rate", f"{row['cache hit rate']:.0%}")
    cols[3].metric("p50 latency", f"{row['p50 latency (ms)'] / 1000:.2f} s")
    cols[4].metric("p95 latency", f"{row['p95 latency (ms)'] / 1000:.2f} s")
    cols[5].metric("Accuracy", "–" if pd.isna(row["accuracy"]) else f"{row['accuracy']:.1%}")

left, right = st.columns(2)

with left:
    st.subheader("How requests were served")
    split = view.groupby(["run", "served_by"]).size().reset_index(name="requests")
    split["share"] = split["requests"] / split.groupby("run")["requests"].transform("sum")
    st.altair_chart(
        alt.Chart(split).mark_bar(cornerRadiusEnd=4, stroke="#fcfcfb", strokeWidth=2).encode(
            y=alt.Y("run:N", title=None, sort=selected),
            x=alt.X("share:Q", title="Share of requests", axis=alt.Axis(format="%", grid=False)),
            color=alt.Color("served_by:N", scale=served_scale, title="Served by",
                            sort=SERVED_ORDER, legend=alt.Legend(orient="top")),
            order=alt.Order("served_by_order:Q"),
            tooltip=["run", "served_by", "requests", alt.Tooltip("share:Q", format=".0%")],
        ).transform_calculate(served_by_order=f"indexof({SERVED_ORDER}, datum.served_by)"),
        width="stretch")

with right:
    st.subheader("Cumulative LLM cost")
    st.altair_chart(
        alt.Chart(view).mark_line(strokeWidth=2).encode(
            x=alt.X("n:Q", title="Request #"),
            y=alt.Y("cumulative_cost:Q", title="Cost so far ($)", axis=alt.Axis(format="$.2f")),
            color=alt.Color("run:N", scale=run_scale, title=None, legend=alt.Legend(orient="top")),
            tooltip=["run", "n", alt.Tooltip("cumulative_cost:Q", format="$.3f"), "served_by"],
        ),
        width="stretch")

left, right = st.columns(2)
by_served = view.groupby(["run", "served_by"]).agg(
    requests=("request_id", "size"), cost_per_request=("cost_usd", "mean"),
    p50_ms=("total_ms", "median"), accuracy=("correct", "mean")).reset_index()

with left:
    st.subheader("Cost per request by path")
    st.altair_chart(
        alt.Chart(by_served).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("cost_per_request:Q", title="Mean cost ($)", axis=alt.Axis(format="$.4f")),
            y=alt.Y("served_by:N", title=None, sort=SERVED_ORDER),
            yOffset=alt.YOffset("run:N", sort=selected),
            color=alt.Color("run:N", scale=run_scale, title=None, legend=alt.Legend(orient="top")),
            tooltip=["run", "served_by", "requests", alt.Tooltip("cost_per_request:Q", format="$.5f")],
        ),
        width="stretch")

with right:
    st.subheader("Accuracy by path")
    judged = by_served.dropna(subset=["accuracy"])
    if judged.empty:
        st.caption("No judged requests in these runs (accuracy comes from eval runs).")
    else:
        st.altair_chart(
            alt.Chart(judged).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("accuracy:Q", title="Judged correct", axis=alt.Axis(format="%"),
                        scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("served_by:N", title=None, sort=SERVED_ORDER),
                yOffset=alt.YOffset("run:N", sort=selected),
                color=alt.Color("run:N", scale=run_scale, title=None, legend=alt.Legend(orient="top")),
                tooltip=["run", "served_by", "requests", alt.Tooltip("accuracy:Q", format=".1%")],
            ),
            width="stretch")

st.subheader("Latency")
st.altair_chart(
    alt.Chart(view).mark_tick(thickness=2, size=18, opacity=0.6).encode(
        x=alt.X("total_ms:Q", title="End-to-end latency (ms)", scale=alt.Scale(type="symlog")),
        y=alt.Y("run:N", title=None, sort=selected),
        color=alt.Color("served_by:N", scale=served_scale, title="Served by", sort=SERVED_ORDER,
                        legend=alt.Legend(orient="top")),
        tooltip=["run", "served_by", alt.Tooltip("total_ms:Q", format=",.0f"), "query"],
    ),
    width="stretch")

st.subheader("Summary table")
st.dataframe(table.style.format({
    "requests": "{:.0f}", "cost / request ($)": "${:.5f}", "total cost ($)": "${:.3f}",
    "cache hit rate": "{:.1%}", "p50 latency (ms)": "{:,.0f}", "p95 latency (ms)": "{:,.0f}",
    "accuracy": "{:.1%}"}), width="stretch")
st.dataframe(by_served, width="stretch", hide_index=True)

st.subheader("Recent requests")
st.dataframe(view.sort_values("ts", ascending=False).head(200)[
    ["ts", "run", "query", "served_by", "route", "cost_usd", "total_ms", "cache_similarity", "correct"]],
    width="stretch", hide_index=True)
