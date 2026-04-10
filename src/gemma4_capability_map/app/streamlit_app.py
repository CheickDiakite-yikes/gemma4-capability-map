from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from gemma4_capability_map.knowledge_work.replay import load_episode_traces
from gemma4_capability_map.metrics.failure_taxonomy import failure_tags
from gemma4_capability_map.reporting.knowledge_work_board import build_intent_comparison_rows
from gemma4_capability_map.traces.replay import load_traces


def main() -> None:
    st.set_page_config(page_title="gemma4-capability-map", layout="wide")
    st.title("gemma4-capability-map")
    st.caption("Failure explorer for reasoning, routing, retrieval, visual tool orchestration, and full-stack Gemma benchmark traces.")
    mode = st.sidebar.selectbox("Explorer mode", ["task_traces", "knowledge_work_episodes", "knowledge_work_board"], index=0)
    if mode == "knowledge_work_board":
        _board_mode()
        return
    if mode == "knowledge_work_episodes":
        _episode_mode()
        return
    default_trace_path = Path("results/raw/traces.jsonl")
    trace_path = Path(st.sidebar.text_input("Trace file", str(default_trace_path)))

    if not trace_path.exists():
        st.info(f"Trace file not found: {trace_path}")
        st.stop()

    traces = load_traces(trace_path)
    if not traces:
        st.warning("No traces found.")
        st.stop()

    rows = [_trace_row(trace) for trace in traces]
    frame = pd.DataFrame(rows)
    st.sidebar.subheader("Filters")
    selected_tracks = st.sidebar.multiselect("Track", sorted(frame["track"].unique()), default=sorted(frame["track"].unique()))
    selected_architectures = st.sidebar.multiselect(
        "Architecture",
        sorted(frame["architecture"].unique()),
        default=sorted(frame["architecture"].unique()),
    )
    selected_tags = st.sidebar.multiselect(
        "Benchmark Tag",
        sorted({tag for tags in frame["benchmark_tags"] for tag in tags}),
        default=[],
    )
    selected_autonomy = st.sidebar.multiselect(
        "Autonomy Level",
        sorted({value for value in frame["autonomy_level"] if value}),
        default=[],
    )
    selected_failures = st.sidebar.multiselect(
        "Failure Tag",
        sorted({tag for tags in frame["failure_tags"] for tag in tags}),
        default=[],
    )
    success_mode = st.sidebar.selectbox("Run type", ["all", "only failures", "only successes"], index=0)

    filtered = frame[frame["track"].isin(selected_tracks) & frame["architecture"].isin(selected_architectures)]
    if selected_tags:
        filtered = filtered[filtered["benchmark_tags"].apply(lambda tags: any(tag in tags for tag in selected_tags))]
    if selected_autonomy:
        filtered = filtered[filtered["autonomy_level"].isin(selected_autonomy)]
    if selected_failures:
        filtered = filtered[filtered["failure_tags"].apply(lambda tags: any(tag in tags for tag in selected_failures))]
    if success_mode == "only failures":
        filtered = filtered[filtered["success"] < 1.0]
    elif success_mode == "only successes":
        filtered = filtered[filtered["success"] >= 1.0]

    if filtered.empty:
        st.warning("No runs match the current filters.")
        st.stop()

    st.subheader("Aggregate View")
    summary = (
        filtered.groupby(["architecture", "track"], as_index=False)
        .agg(
            runs=("run_id", "count"),
            success_rate=("success", "mean"),
            avg_interface_reliability=("interface_reliability_score", "mean"),
            avg_steps=("steps_taken", "mean"),
        )
        .sort_values(["track", "architecture"])
    )
    st.dataframe(summary, use_container_width=True)

    chart_metric = st.selectbox("Chart metric", ["success", "interface_reliability_score", "steps_taken"], index=0)
    chart = (
        filtered.groupby(["architecture", "track"], as_index=False)[chart_metric]
        .mean()
        .pivot(index="track", columns="architecture", values=chart_metric)
        .fillna(0.0)
    )
    st.bar_chart(chart)

    st.subheader("Run Table")
    st.dataframe(
        filtered[
            [
                "run_id",
                "track",
                "architecture",
                "backend",
                "success",
                "interface_reliability_score",
                "steps_taken",
                "real_world_readiness_score",
                "autonomy_level",
                "risk_tier",
                "failure_summary",
                "language",
                "schema",
                "context",
                "efficiency",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    selected_run_id = st.selectbox("Run", filtered["run_id"].tolist())
    selected = next(trace for trace in traces if trace.run_id == selected_run_id)
    st.subheader("Run Overview")
    st.json(
        {
            "task_id": selected.task_id,
            "variant_id": selected.variant_id,
            "track": selected.track.value,
            "architecture": selected.architecture,
            "backend": selected.backend,
            "thinking_enabled": selected.thinking_enabled,
            "stressors": selected.stressors,
            "benchmark_tags": selected.benchmark_tags,
            "real_world_profile": selected.real_world_profile.model_dump(mode="json") if selected.real_world_profile else None,
            "metrics": selected.metrics,
        }
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Prompt Artifacts")
        st.json(selected.prompt_artifacts)
    with right:
        st.subheader("Failure Tags")
        st.write(", ".join(failure_tags(selected)) or "none")

    trace_left, trace_right = st.columns(2)
    with trace_left:
        st.subheader("Planning Raw Outputs")
        st.json(selected.prompt_artifacts.get("planning_raw_outputs", []))
    with trace_right:
        st.subheader("Tool Steps")
        st.json([step.model_dump(mode="json") for step in selected.tool_steps])

    st.subheader("Retrieval Hits")
    st.json([hit.model_dump(mode="json") for hit in selected.retrieval_hits])

    st.subheader("State Transitions")
    st.json([transition.model_dump(mode="json") for transition in selected.state_transitions])

    st.subheader("Final Answer")
    st.write(selected.final_answer)


def _episode_mode() -> None:
    default_trace_path = Path("results/knowledge_work/replayable_core/episode_traces.jsonl")
    trace_path = Path(st.sidebar.text_input("Episode trace file", str(default_trace_path)))

    if not trace_path.exists():
        st.info(f"Episode trace file not found: {trace_path}")
        st.stop()

    traces = load_episode_traces(trace_path)
    if not traces:
        st.warning("No episode traces found.")
        st.stop()

    rows = [_episode_row(trace) for trace in traces]
    frame = pd.DataFrame(rows)
    selected_roles = st.sidebar.multiselect("Role Family", sorted(frame["role_family"].unique()), default=sorted(frame["role_family"].unique()))
    selected_lanes = st.sidebar.multiselect("Lane", sorted(frame["lane"].unique()), default=sorted(frame["lane"].unique()))
    filtered = frame[frame["role_family"].isin(selected_roles) & frame["lane"].isin(selected_lanes)]
    if filtered.empty:
        st.warning("No episodes match the current filters.")
        st.stop()

    st.subheader("Episode Aggregate View")
    summary = (
        filtered.groupby(["role_family", "lane"], as_index=False)
        .agg(
            runs=("run_id", "count"),
            artifact_quality=("artifact_quality_score", "mean"),
            strict_interface=("strict_interface_score", "mean"),
            recovered_execution=("recovered_execution_score", "mean"),
            readiness=("role_readiness_score", "mean"),
        )
        .sort_values(["role_family", "lane"])
    )
    st.dataframe(summary, use_container_width=True)

    st.subheader("Episode Table")
    st.dataframe(
        filtered[
            [
                "run_id",
                "episode_id",
                "role_family",
                "lane",
                "artifact_quality_score",
                "strict_interface_score",
                "recovered_execution_score",
                "escalation_correctness",
                "role_readiness_score",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    selected_run_id = st.selectbox("Episode run", filtered["run_id"].tolist())
    selected = next(trace for trace in traces if trace.run_id == selected_run_id)
    st.subheader("Episode Overview")
    st.json(
        {
            "episode_id": selected.episode_id,
            "role_family": selected.role_family.value,
            "lane": selected.lane.value,
            "workspace_id": selected.workspace_id,
            "benchmark_tags": selected.benchmark_tags,
            "scorecard": selected.scorecard.model_dump(mode="json"),
        }
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Stage Timeline")
        st.json([stage.model_dump(mode="json") for stage in selected.stage_traces])
    with right:
        st.subheader("Browser Replay Summary")
        st.json([action.model_dump(mode="json") for action in selected.browser_actions])

    artifact_frame = pd.DataFrame(
        [
            {
                "artifact_id": version.artifact_id,
                "revision": version.revision,
                "source_stage": version.source_stage,
                "score": version.score,
                "content": version.content,
            }
            for version in selected.artifact_versions
        ]
    )
    st.subheader("Artifact Revisions")
    st.dataframe(artifact_frame, use_container_width=True, hide_index=True)

    st.subheader("Memory Updates")
    st.json([update.model_dump(mode="json") for update in selected.memory_updates])

    st.subheader("Review History")
    st.json([review.model_dump(mode="json") for review in selected.review_history])


def _board_mode() -> None:
    default_board_path = Path("results/history/knowledge_work_board_latest.csv")
    board_path = Path(st.sidebar.text_input("Board CSV", str(default_board_path)))

    if not board_path.exists():
        st.info(f"Board file not found: {board_path}")
        st.stop()

    frame = pd.read_csv(board_path)
    if frame.empty:
        st.warning("No board rows found.")
        st.stop()
    for column in (
        "capability_family",
        "executor_mode",
        "modality",
        "run_scope",
        "total_cost_per_mtok",
        "warmup_load_ms",
        "last_request_elapsed_ms",
        "requests_completed",
        "role_breakdown_json",
        "category_breakdown_json",
        "track_breakdown_json",
    ):
        if column not in frame.columns:
            frame[column] = ""

    lane_values = sorted(frame["lane"].dropna().unique().tolist())
    intent_values = sorted(frame["run_intent"].dropna().unique().tolist())
    scope_values = sorted(value for value in frame.get("run_scope", pd.Series(dtype=str)).dropna().unique().tolist() if value)
    provider_values = sorted(frame["provider"].dropna().unique().tolist())
    capability_values = sorted(value for value in frame.get("capability_family", pd.Series(dtype=str)).dropna().unique().tolist() if value)
    modality_values = sorted(value for value in frame.get("modality", pd.Series(dtype=str)).dropna().unique().tolist() if value)
    executor_values = sorted(value for value in frame.get("executor_mode", pd.Series(dtype=str)).dropna().unique().tolist() if value)

    selected_lanes = st.sidebar.multiselect("Lane", lane_values, default=lane_values)
    selected_intents = st.sidebar.multiselect("Run intent", intent_values, default=intent_values)
    selected_scopes = st.sidebar.multiselect("Run scope", scope_values, default=scope_values)
    selected_providers = st.sidebar.multiselect("Provider", provider_values, default=provider_values)
    selected_capabilities = st.sidebar.multiselect("Capability family", capability_values, default=capability_values)
    selected_modalities = st.sidebar.multiselect("Modality", modality_values, default=modality_values)
    selected_executors = st.sidebar.multiselect("Executor mode", executor_values, default=executor_values)
    local_mode = st.sidebar.selectbox("Deployment", ["all", "local only", "non-local only"], index=0)
    rank_metric = st.sidebar.selectbox(
        "Rank by",
        [
            "real_world_readiness_avg",
            "strict_interface_avg",
            "browser_workflow_avg",
            "artifact_quality_avg",
            "recovered_execution_avg",
            "escalation_correctness_avg",
        ],
        index=0,
    )

    filtered = frame[
        frame["lane"].isin(selected_lanes)
        & frame["run_intent"].isin(selected_intents)
        & frame["run_scope"].fillna("").isin(selected_scopes or scope_values or [""])
        & frame["provider"].isin(selected_providers)
        & frame["capability_family"].fillna("").isin(selected_capabilities or capability_values or [""])
        & frame["modality"].fillna("").isin(selected_modalities or modality_values or [""])
        & frame["executor_mode"].fillna("").isin(selected_executors or executor_values or [""])
    ]
    if local_mode == "local only":
        filtered = filtered[filtered["local"] == True]  # noqa: E712
    elif local_mode == "non-local only":
        filtered = filtered[filtered["local"] == False]  # noqa: E712

    if filtered.empty:
        st.warning("No board rows match the current filters.")
        st.stop()

    ranked = filtered.sort_values([rank_metric, "strict_interface_avg", "browser_workflow_avg"], ascending=[False, False, False]).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    st.subheader("Benchmark Leaderboard")
    st.dataframe(
        ranked[
            [
                "rank",
                "display_name",
                "lane",
                "run_intent",
                "run_scope",
                "capability_family",
                "executor_mode",
                "modality",
                "episode_count",
                "pass_count",
                "refine_count",
                "fail_count",
                "real_world_readiness_avg",
                "strict_interface_avg",
                "browser_workflow_avg",
                "artifact_quality_avg",
                "recovered_execution_avg",
                "escalation_correctness_avg",
                "total_params_b",
                "warmup_load_ms",
                "last_request_elapsed_ms",
                "total_cost_per_mtok",
                "input_cost_per_mtok",
                "provider",
                "local",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    x_axis = st.selectbox(
        "Scatter X",
        [
            "total_params_b",
            "reasoner_params_b",
            "episode_count",
            "warmup_load_ms",
            "last_request_elapsed_ms",
            "input_cost_per_mtok",
            "output_cost_per_mtok",
            "total_cost_per_mtok",
        ],
        index=0,
    )
    y_axis = st.selectbox(
        "Scatter Y",
        [
            "real_world_readiness_avg",
            "strict_interface_avg",
            "browser_workflow_avg",
            "artifact_quality_avg",
            "recovered_execution_avg",
            "escalation_correctness_avg",
        ],
        index=0,
    )

    scatter = ranked.dropna(subset=[x_axis, y_axis])
    if scatter.empty:
        st.info("No rows have values for the selected scatter axes.")
    else:
        chart = (
            alt.Chart(scatter)
            .mark_circle(size=140)
            .encode(
                x=alt.X(x_axis, title=x_axis.replace("_", " ").title()),
                y=alt.Y(y_axis, title=y_axis.replace("_", " ").title(), scale=alt.Scale(domain=[0, 1.02])),
                color=alt.Color("lane:N", title="Lane"),
                shape=alt.Shape("run_intent:N", title="Run Intent"),
                tooltip=[
                    "display_name",
                    "lane",
                    "run_intent",
                    "episode_count",
                    "pass_count",
                    "refine_count",
                    "fail_count",
                    "real_world_readiness_avg",
                    "strict_interface_avg",
                    "browser_workflow_avg",
                    "artifact_quality_avg",
                    "recovered_execution_avg",
                    "total_params_b",
                    "warmup_load_ms",
                    "last_request_elapsed_ms",
                    "total_cost_per_mtok",
                ],
            )
            .interactive()
        )
        st.subheader("Performance Scatter")
        st.altair_chart(chart, use_container_width=True)

    role_breakdown = _flatten_breakdown_frame(ranked, "role_breakdown_json", "role_family")
    if not role_breakdown.empty:
        st.subheader("Role Breakdown")
        st.dataframe(role_breakdown, use_container_width=True, hide_index=True)
        _render_breakdown_status_chart(role_breakdown, "role_family", "Role Status Breakdown")

    category_breakdown = _flatten_breakdown_frame(ranked, "category_breakdown_json", "category")
    if not category_breakdown.empty:
        st.subheader("Category Breakdown")
        st.dataframe(category_breakdown, use_container_width=True, hide_index=True)
        _render_breakdown_status_chart(category_breakdown, "category", "Category Status Breakdown")

    track_breakdown = _flatten_breakdown_frame(ranked, "track_breakdown_json", "track_tag")
    if not track_breakdown.empty:
        st.subheader("Track Breakdown")
        st.dataframe(track_breakdown, use_container_width=True, hide_index=True)
        _render_breakdown_status_chart(track_breakdown, "track_tag", "Track Status Breakdown")

    comparison_rows = build_intent_comparison_rows(filtered.to_dict(orient="records"))
    comparison_frame = pd.DataFrame(comparison_rows)
    if not comparison_frame.empty:
        st.subheader("Canonical vs Exploratory Comparison")
        st.caption("Canonical and exploratory runs are shown side-by-side for each system and lane.")
        st.dataframe(
            comparison_frame[
                [
                    "display_name",
                    "lane",
                    "canonical_readiness",
                    "exploratory_readiness",
                    "readiness_delta",
                    "canonical_strict_interface",
                    "exploratory_strict_interface",
                    "strict_delta",
                    "canonical_browser_workflow",
                    "exploratory_browser_workflow",
                    "browser_delta",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        delta_chart = (
            alt.Chart(comparison_frame)
            .mark_bar()
            .encode(
                x=alt.X("display_name:N", sort="-y", title="System"),
                y=alt.Y("readiness_delta:Q", title="Exploratory minus Canonical Readiness"),
                color=alt.Color("capability_family:N", title="Capability Family"),
                tooltip=[
                    "display_name",
                    "lane",
                    "capability_family",
                    "canonical_readiness",
                    "exploratory_readiness",
                    "readiness_delta",
                    "canonical_strict_interface",
                    "exploratory_strict_interface",
                    "strict_delta",
                    "canonical_browser_workflow",
                    "exploratory_browser_workflow",
                    "browser_delta",
                ],
            )
        )
        st.altair_chart(delta_chart, use_container_width=True)

    selected_label = st.selectbox("System", ranked["display_name"].tolist())
    selected = ranked.loc[ranked["display_name"] == selected_label].iloc[0].to_dict()
    st.subheader("System Overview")
    st.json(selected)


def _flatten_breakdown_frame(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, record in frame.iterrows():
        raw = record.get(column, "")
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for key, values in payload.items():
            if not isinstance(values, dict):
                continue
            rows.append(
                {
                    "display_name": record.get("display_name", ""),
                    "lane": record.get("lane", ""),
                    "run_intent": record.get("run_intent", ""),
                    label: key,
                    **values,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["display_name", label]).reset_index(drop=True)


def _render_breakdown_status_chart(frame: pd.DataFrame, label: str, title: str) -> None:
    required = {"pass_count", "refine_count", "fail_count", label}
    if frame.empty or not required.issubset(frame.columns):
        return
    counts = frame[[label, "pass_count", "refine_count", "fail_count"]].copy()
    melted = counts.melt(id_vars=[label], var_name="status", value_name="count")
    grouped = melted.groupby([label, "status"], as_index=False)["count"].sum()
    chart = (
        alt.Chart(grouped)
        .mark_bar()
        .encode(
            x=alt.X(f"{label}:N", sort="-y", title=label.replace("_", " ").title()),
            y=alt.Y("count:Q", title="Episode Count"),
            color=alt.Color("status:N", title="Status"),
            tooltip=[label, "status", "count"],
        )
    )
    st.altair_chart(chart, use_container_width=True)


def _trace_row(trace):
    tags = failure_tags(trace)
    return {
        "run_id": trace.run_id,
        "task_id": trace.task_id,
        "track": trace.track.value,
        "architecture": trace.architecture,
        "backend": trace.backend,
        "success": float(trace.metrics.get("success", 0.0)),
        "interface_reliability_score": float(trace.metrics.get("interface_reliability_score", 0.0)),
        "steps_taken": int(trace.metrics.get("steps_taken", 0)),
        "real_world_readiness_score": float(trace.metrics.get("real_world_readiness_score", 0.0)),
        "failure_tags": tags,
        "failure_summary": ", ".join(tags) if tags else "none",
        "benchmark_tags": trace.benchmark_tags,
        "autonomy_level": trace.real_world_profile.autonomy_level if trace.real_world_profile else None,
        "risk_tier": trace.real_world_profile.risk_tier if trace.real_world_profile else None,
        "language": trace.stressors.get("language"),
        "schema": trace.stressors.get("schema"),
        "context": trace.stressors.get("context"),
        "efficiency": trace.stressors.get("efficiency"),
    }


def _episode_row(trace):
    return {
        "run_id": trace.run_id,
        "episode_id": trace.episode_id,
        "role_family": trace.role_family.value,
        "lane": trace.lane.value,
        "artifact_quality_score": trace.scorecard.artifact_quality_score,
        "strict_interface_score": trace.scorecard.strict_interface_score,
        "recovered_execution_score": trace.scorecard.recovered_execution_score,
        "escalation_correctness": trace.scorecard.escalation_correctness,
        "role_readiness_score": trace.scorecard.role_readiness_score,
    }

if __name__ == "__main__":
    main()
