from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from gemma4_capability_map.app.theme import inject_theme, pill_html
from gemma4_capability_map.app.view_models import build_console_snapshot
from gemma4_capability_map.app.views import render_mobile_companion, render_operator_console
from gemma4_capability_map.knowledge_work.replay import load_episode_traces
from gemma4_capability_map.metrics.failure_taxonomy import failure_tags
from gemma4_capability_map.reporting.knowledge_work_board import (
    build_comparison_batch_rows,
    build_intent_comparison_rows,
    build_lane_summary_rows,
    build_leaderboard_rows,
    build_public_summary,
    build_runtime_profile_rows,
)
from gemma4_capability_map.runtime.core import LocalAgentRuntime
from gemma4_capability_map.traces.replay import load_traces


def main() -> None:
    st.set_page_config(page_title="Moonie", layout="wide")
    mode = st.sidebar.selectbox(
        "Surface",
        [
            "operator_console",
            "mobile_companion",
            "knowledge_work_board",
            "knowledge_work_episodes",
            "task_traces",
        ],
        index=0,
    )
    theme_mode = "mobile" if mode == "mobile_companion" else "board" if mode == "knowledge_work_board" else "desktop"
    inject_theme(theme_mode)
    if mode == "operator_console":
        render_operator_console(_runtime())
        return
    if mode == "mobile_companion":
        render_mobile_companion(_runtime())
        return
    st.title("Moonie")
    st.caption("Local-first benchmark explorer, operator console, and runtime shell for agent sessions.")
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


@st.cache_resource
def _runtime() -> LocalAgentRuntime:
    return LocalAgentRuntime()


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
    external_benchmark_path = board_path.parent / "knowledge_work_external_benchmarks.csv"
    external_benchmark_summary_path = board_path.parent / "knowledge_work_external_benchmark_summary.json"

    if not board_path.exists():
        st.info(f"Board file not found: {board_path}")
        st.stop()

    frame = pd.read_csv(board_path)
    if frame.empty:
        st.warning("No board rows found.")
        st.stop()
    for boolean_column in ("local", "publishable_default"):
        if boolean_column in frame.columns:
            frame[boolean_column] = frame[boolean_column].astype(str).str.lower().eq("true")
    for column in (
        "capability_family",
        "executor_mode",
        "modality",
        "comparison_tier",
        "publishable_default",
        "run_scope",
        "result_family",
        "comparison_batch",
        "rank_group",
        "board_status",
        "progress_status",
        "validation_error",
        "returncode",
        "timed_out",
        "completed_runs_observed",
        "planned_runs",
        "coverage_ratio",
        "coverage_pct",
        "matrix_complete",
        "full_lane_complete",
        "failure_excerpt",
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
    result_family_values = sorted(value for value in frame.get("result_family", pd.Series(dtype=str)).dropna().unique().tolist() if value)
    comparison_batch_values = sorted(value for value in frame.get("comparison_batch", pd.Series(dtype=str)).dropna().unique().tolist() if value)
    capability_values = sorted(value for value in frame.get("capability_family", pd.Series(dtype=str)).dropna().unique().tolist() if value)
    comparison_tier_values = sorted(value for value in frame.get("comparison_tier", pd.Series(dtype=str)).dropna().unique().tolist() if value)
    modality_values = sorted(value for value in frame.get("modality", pd.Series(dtype=str)).dropna().unique().tolist() if value)
    executor_values = sorted(value for value in frame.get("executor_mode", pd.Series(dtype=str)).dropna().unique().tolist() if value)
    board_status_values = sorted(value for value in frame.get("board_status", pd.Series(dtype=str)).dropna().unique().tolist() if value)

    selected_lanes = st.sidebar.multiselect("Lane", lane_values, default=lane_values)
    selected_intents = st.sidebar.multiselect("Run intent", intent_values, default=intent_values)
    selected_scopes = st.sidebar.multiselect("Run scope", scope_values, default=scope_values)
    selected_providers = st.sidebar.multiselect("Provider", provider_values, default=provider_values)
    selected_result_families = st.sidebar.multiselect("Result family", result_family_values, default=result_family_values)
    selected_comparison_batches = st.sidebar.multiselect("Comparison batch", comparison_batch_values, default=comparison_batch_values)
    selected_capabilities = st.sidebar.multiselect("Capability family", capability_values, default=capability_values)
    selected_comparison_tiers = st.sidebar.multiselect("Comparison tier", comparison_tier_values, default=comparison_tier_values)
    selected_modalities = st.sidebar.multiselect("Modality", modality_values, default=modality_values)
    selected_executors = st.sidebar.multiselect("Executor mode", executor_values, default=executor_values)
    selected_statuses = st.sidebar.multiselect("Run health", board_status_values, default=board_status_values)
    local_mode = st.sidebar.selectbox("Deployment", ["all", "local only", "non-local only"], index=0)
    publishable_only = st.sidebar.checkbox("Publishable defaults only", value=False)
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
        & frame["result_family"].fillna("").isin(selected_result_families or result_family_values or [""])
        & frame["comparison_batch"].fillna("").isin(selected_comparison_batches or comparison_batch_values or [""])
        & frame["capability_family"].fillna("").isin(selected_capabilities or capability_values or [""])
        & frame["comparison_tier"].fillna("").isin(selected_comparison_tiers or comparison_tier_values or [""])
        & frame["modality"].fillna("").isin(selected_modalities or modality_values or [""])
        & frame["executor_mode"].fillna("").isin(selected_executors or executor_values or [""])
        & frame["board_status"].fillna("").isin(selected_statuses or board_status_values or [""])
    ]
    if local_mode == "local only":
        filtered = filtered[filtered["local"] == True]  # noqa: E712
    elif local_mode == "non-local only":
        filtered = filtered[filtered["local"] == False]  # noqa: E712
    if publishable_only:
        filtered = filtered[filtered["publishable_default"] == True]  # noqa: E712

    if filtered.empty:
        st.warning("No board rows match the current filters.")
        st.stop()

    ranked = filtered.sort_values([rank_metric, "strict_interface_avg", "browser_workflow_avg"], ascending=[False, False, False]).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    filtered_rows = filtered.to_dict(orient="records")
    leaderboard_frame = pd.DataFrame(build_leaderboard_rows(filtered_rows))
    lane_summary_frame = pd.DataFrame(build_lane_summary_rows(filtered_rows))
    runtime_profile_frame = pd.DataFrame(build_runtime_profile_rows(filtered_rows))
    comparison_batch_frame = pd.DataFrame(build_comparison_batch_rows(filtered_rows))
    health_frame = filtered.copy()
    public_summary = build_public_summary(filtered_rows)
    workflow_snapshot = build_console_snapshot(_runtime(), board_path)
    external_benchmark_frame = (
        pd.read_csv(external_benchmark_path) if external_benchmark_path.exists() else pd.DataFrame()
    )
    external_benchmark_summary = (
        json.loads(external_benchmark_summary_path.read_text(encoding="utf-8"))
        if external_benchmark_summary_path.exists()
        else {}
    )

    top_local = public_summary.get("highest_readiness_local_profile", {})
    fastest_local = public_summary.get("fastest_local_profile", {})
    efficient_local = public_summary.get("most_efficient_local_profile", {})
    comparison_health = public_summary.get("comparison_health", {})

    st.markdown(
        f"""
        <div class="board-hero">
          <p class="board-title">Knowledge Work Benchmark Board</p>
          <p class="board-subtitle">Publication-grade local-first benchmark surface across full-lane runs, runtime posture, workflow packaging, and category breakdowns.</p>
          <div class="board-summary-grid">
            <div class="board-summary-card">
              <div class="board-summary-label">Latest runs</div>
              <div class="board-summary-value">{public_summary.get('latest_runs', 0)}</div>
              <div class="board-summary-note">Filtered board rows on the current lane and intent slice · {len(public_summary.get('comparison_batches', []))} comparison batches · coverage {(comparison_health.get('avg_coverage', 0.0) or 0.0) * 100:.1f}%.</div>
            </div>
            <div class="board-summary-card">
              <div class="board-summary-label">Publishable profiles</div>
              <div class="board-summary-value">{public_summary.get('publishable_profiles', 0)}</div>
              <div class="board-summary-note">Default headline profiles with full-lane complete coverage in this board view.</div>
            </div>
            <div class="board-summary-card">
              <div class="board-summary-label">Top local</div>
              <div class="board-summary-value">{top_local.get('display_name', 'n/a')}</div>
              <div class="board-summary-note">Readiness {(top_local.get('avg_readiness', 0.0) or 0.0) * 100:.1f}%</div>
            </div>
            <div class="board-summary-card">
              <div class="board-summary-label">Fastest local</div>
              <div class="board-summary-value">{fastest_local.get('display_name', 'n/a')}</div>
              <div class="board-summary-note">{(fastest_local.get('value', 0.0) or 0.0):.0f} ms · cheapest {(efficient_local.get('value', 0.0) or 0.0):.2f} $/Mtok</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview_tab, leaderboard_tab, runtime_tab, workflow_tab, external_tab, breakdown_tab, delta_tab = st.tabs(
        ["Overview", "Leaderboard", "Runtime Profiles", "Workflow Shelf", "External Context", "Breakdowns", "Deltas"]
    )

    with overview_tab:
        if not health_frame.empty:
            st.subheader("Comparison Health")
            health_summary = (
                health_frame.groupby("board_status", as_index=False)
                .agg(
                    systems=("display_name", "count"),
                    avg_coverage=("coverage_ratio", "mean"),
                    avg_readiness=("real_world_readiness_avg", "mean"),
                )
                .sort_values(["systems", "avg_coverage"], ascending=[False, False])
            )
            st.dataframe(health_summary, use_container_width=True, hide_index=True)
            health_chart = (
                alt.Chart(health_summary)
                .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
                .encode(
                    x=alt.X("board_status:N", title="Run health"),
                    y=alt.Y("systems:Q", title="Systems"),
                    color=alt.Color("board_status:N", title="Run health"),
                    tooltip=["board_status", "systems", "avg_coverage", "avg_readiness"],
                )
            )
            st.altair_chart(health_chart, use_container_width=True)
        headline_rows = pd.DataFrame(public_summary.get("headline_systems", []))
        if not headline_rows.empty:
            st.subheader("Headline Systems")
            st.dataframe(headline_rows, use_container_width=True, hide_index=True)
        if not lane_summary_frame.empty:
            st.subheader("Lane Summary")
            st.dataframe(
                lane_summary_frame[
                    [
                        "lane",
                        "run_intent",
                        "systems",
                        "local_systems",
                        "publishable_systems",
                        "avg_coverage",
                        "avg_readiness",
                        "avg_strict_interface",
                        "avg_browser_workflow",
                        "avg_artifact_quality",
                        "total_episodes",
                        "total_pass_count",
                        "total_refine_count",
                        "total_fail_count",
                        "best_display_name",
                        "best_readiness",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
        if not comparison_batch_frame.empty:
            st.subheader("Comparison Batches")
            st.dataframe(
                comparison_batch_frame[
                    [
                        "comparison_batch",
                        "completed_systems",
                        "partial_systems",
                        "timed_out_systems",
                        "failed_systems",
                        "result_families",
                        "systems",
                        "lanes",
                        "run_intents",
                        "observed_episodes",
                        "total_episodes",
                        "coverage_ratio",
                        "matrix_complete",
                        "avg_readiness",
                        "avg_strict_interface",
                        "avg_browser_workflow",
                        "best_display_name",
                        "best_readiness",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
            batch_chart = (
                alt.Chart(comparison_batch_frame)
                .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
                .encode(
                    x=alt.X("comparison_batch:N", sort="-y", title="Comparison Batch"),
                    y=alt.Y("avg_readiness:Q", title="Average Readiness", scale=alt.Scale(domain=[0, 1.02])),
                    color=alt.Color("result_families:N", title="Result Family"),
                    tooltip=[
                        "comparison_batch",
                        "completed_systems",
                        "partial_systems",
                        "timed_out_systems",
                        "failed_systems",
                        "result_families",
                        "systems",
                        "lanes",
                        "run_intents",
                        "avg_readiness",
                        "best_display_name",
                        "best_readiness",
                    ],
                )
            )
            st.altair_chart(batch_chart, use_container_width=True)
        if not lane_summary_frame.empty:
            lane_chart = (
                alt.Chart(lane_summary_frame)
                .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
                .encode(
                x=alt.X("lane:N", title="Lane"),
                y=alt.Y("avg_readiness:Q", title="Average Readiness", scale=alt.Scale(domain=[0, 1.02])),
                color=alt.Color("run_intent:N", title="Run Intent"),
                tooltip=[
                    "lane",
                    "run_intent",
                    "systems",
                    "local_systems",
                    "avg_readiness",
                    "avg_strict_interface",
                    "avg_browser_workflow",
                    "best_display_name",
                    ],
                )
            )
            st.altair_chart(lane_chart, use_container_width=True)

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

    with leaderboard_tab:
        st.subheader("System Leaderboard")
        st.caption("Ranks are computed within each lane and run-intent bucket, and only completed runs are ranked.")
        board_leaderboard = leaderboard_frame if not leaderboard_frame.empty else ranked
        st.dataframe(
            board_leaderboard[
                [
                    "rank",
                    "rank_group",
                    "display_name",
                    "lane",
                    "run_intent",
                    "run_scope",
                    "result_family",
                    "comparison_batch",
                    "comparison_tier",
                    "publishable_default",
                    "capability_family",
                    "executor_mode",
                    "modality",
                    "episode_count",
                    "pass_count",
                    "refine_count",
                    "fail_count",
                    "coverage_ratio",
                    "full_lane_complete",
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
                    "provider",
                    "local",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        incomplete = health_frame[health_frame["board_status"] != "completed"]
        if not incomplete.empty:
            st.subheader("Incomplete Or Failed Comparison Rows")
            st.dataframe(
                incomplete[
                    [
                        "display_name",
                        "lane",
                        "run_intent",
                        "board_status",
                        "completed_runs_observed",
                        "planned_runs",
                        "coverage_ratio",
                        "validation_error",
                        "failure_excerpt",
                        "comparison_batch",
                        "output_dir",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    with runtime_tab:
        if runtime_profile_frame.empty:
            st.info("No runtime profiles available for the current filter set.")
        else:
            st.subheader("Runtime Profile View")
            st.dataframe(
                runtime_profile_frame[
                    [
                        "display_name",
                        "run_intent",
                        "comparison_tier",
                        "publishable_default",
                        "capability_family",
                        "executor_mode",
                        "modality",
                        "lane_count",
                        "lanes",
                        "avg_readiness",
                        "avg_strict_interface",
                        "avg_browser_workflow",
                        "avg_artifact_quality",
                        "avg_recovered_execution",
                        "episode_count",
                        "pass_count",
                        "refine_count",
                        "fail_count",
                        "total_params_b",
                        "warmup_load_ms",
                        "last_request_elapsed_ms",
                        "requests_completed",
                        "total_cost_per_mtok",
                        "best_lane",
                        "best_lane_readiness",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
            runtime_chart = (
                alt.Chart(runtime_profile_frame)
                .mark_circle(size=180)
                .encode(
                    x=alt.X("last_request_elapsed_ms:Q", title="Last request elapsed (ms)"),
                    y=alt.Y("avg_readiness:Q", title="Average readiness", scale=alt.Scale(domain=[0, 1.02])),
                    color=alt.Color("capability_family:N", title="Capability family"),
                    shape=alt.Shape("executor_mode:N", title="Executor mode"),
                    tooltip=[
                        "display_name",
                        "run_intent",
                        "avg_readiness",
                        "avg_strict_interface",
                        "avg_browser_workflow",
                        "total_params_b",
                        "last_request_elapsed_ms",
                        "total_cost_per_mtok",
                    ],
                )
            )
            st.altair_chart(runtime_chart, use_container_width=True)
            params_chart = (
                alt.Chart(runtime_profile_frame)
                .mark_circle(size=180)
                .encode(
                    x=alt.X("total_params_b:Q", title="Total Parameters (B)"),
                    y=alt.Y("avg_readiness:Q", title="Average readiness", scale=alt.Scale(domain=[0, 1.02])),
                    color=alt.Color("executor_mode:N", title="Executor mode"),
                    shape=alt.Shape("run_intent:N", title="Run intent"),
                    tooltip=[
                        "display_name",
                        "run_intent",
                        "capability_family",
                        "avg_readiness",
                        "total_params_b",
                        "last_request_elapsed_ms",
                        "total_cost_per_mtok",
                    ],
                )
            )
            st.altair_chart(params_chart, use_container_width=True)

    with workflow_tab:
        workflow_cards_by_lane = workflow_snapshot.get("workflow_cards_by_lane", {})
        workflow_cards = workflow_snapshot["workflow_cards"]
        workflow_frame = pd.DataFrame(workflow_cards)
        if workflow_frame.empty and not workflow_cards_by_lane:
            st.info("No packaged workflows available.")
        else:
            st.subheader("Packaged Workflow Lens")
            workflow_tabs = st.tabs(["Replayable workflows", "Live workflows"])
            for tab, lane_key, lane_label in zip(
                workflow_tabs,
                ["replayable_core", "live_web_stress"],
                ["Replayable", "Live"],
                strict=False,
            ):
                lane_cards = workflow_cards_by_lane.get(lane_key, [])
                with tab:
                    lane_frame = pd.DataFrame(lane_cards)
                    if lane_frame.empty:
                        st.info(f"No {lane_label.lower()} packaged workflows available.")
                        continue
                    st.dataframe(
                        lane_frame[
                            [
                                "title",
                                "role_family",
                                "category",
                                "lane",
                                "supports_approval",
                                "recommended_short_label",
                                "recommended_readiness",
                                "active_sessions",
                                "pending_approvals",
                                "completed_sessions",
                                "latest_status",
                                "latest_artifact_count",
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.markdown('<div class="breakdown-shell">', unsafe_allow_html=True)
                    st.caption(f"{lane_label} workflow shelf keeps benchmark-backed packaging visible alongside the system leaderboard.")
                    for workflow in lane_cards[:4]:
                        st.markdown(
                            f"""
                            <div class="console-card">
                              <div class="console-row">
                                <div>
                                  <h4>{workflow['title']}</h4>
                                  <div class="console-muted">{workflow['subtitle']}</div>
                                  <div class="console-muted">Recommended {workflow['recommended_short_label']} · readiness {(workflow['recommended_readiness'] or 0.0) * 100:.1f}%</div>
                                </div>
                                {pill_html('review' if workflow['supports_approval'] else 'fast path', 'awaiting_approval' if workflow['supports_approval'] else 'completed')}
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

    with external_tab:
        _render_external_benchmark_context(external_benchmark_frame, external_benchmark_summary)

    with breakdown_tab:
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

    with delta_tab:
        comparison_rows = build_intent_comparison_rows(filtered_rows)
        comparison_frame = pd.DataFrame(comparison_rows)
        if not comparison_frame.empty:
            st.subheader("Canonical vs Exploratory Comparison")
            st.caption("Canonical and exploratory runs are shown side-by-side for each system and lane.")
            delta_metrics = {
                "Readiness Δ": comparison_frame["readiness_delta"].mean(),
                "Strict Δ": comparison_frame["strict_delta"].mean(),
                "Browser Δ": comparison_frame["browser_delta"].mean(),
            }
            delta_cols = st.columns(3, gap="small")
            for column, (label, value) in zip(delta_cols, delta_metrics.items(), strict=False):
                with column:
                    st.metric(label, f"{(value or 0.0) * 100:+.1f} pts")
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


def _render_external_benchmark_context(frame: pd.DataFrame, summary: dict[str, object]) -> None:
    st.subheader("Published External Benchmark Context")
    st.caption(
        "These rows are published external benchmark results from official sources. They are not Moonie-reproduced runs and should be read as context, not same-harness leaderboard rows."
    )
    if frame.empty:
        st.info("No published external benchmark rows are available in the current history directory.")
        return

    provider_values = sorted(value for value in frame.get("provider", pd.Series(dtype=str)).dropna().unique().tolist() if value)
    group_values = sorted(value for value in frame.get("benchmark_group", pd.Series(dtype=str)).dropna().unique().tolist() if value)
    selected_providers = st.multiselect("External providers", provider_values, default=provider_values, key="external_providers")
    selected_groups = st.multiselect("Benchmark groups", group_values, default=group_values, key="external_groups")
    filtered = frame[
        frame["provider"].isin(selected_providers or provider_values or [""])
        & frame["benchmark_group"].isin(selected_groups or group_values or [""])
    ].copy()

    if filtered.empty:
        st.warning("No external benchmark rows match the current provider/group filters.")
        return

    models = int(summary.get("model_count", filtered["model_id"].nunique()) or 0)
    rows = int(summary.get("row_count", len(frame)) or 0)
    latest_date = str(summary.get("latest_published_date", "") or "n/a")
    metric_cols = st.columns(3, gap="small")
    metric_cols[0].metric("Tracked models", str(models))
    metric_cols[1].metric("Published rows", str(rows))
    metric_cols[2].metric("Latest source date", latest_date)

    benchmark_counts = pd.DataFrame(summary.get("benchmark_counts", []))
    if not benchmark_counts.empty:
        st.subheader("Benchmark Coverage")
        st.dataframe(benchmark_counts, use_container_width=True, hide_index=True)

    st.subheader("Published External Rows")
    st.dataframe(
        filtered[
            [
                "display_name",
                "provider",
                "benchmark",
                "benchmark_group",
                "score_display",
                "published_date",
                "source_scope",
                "source_kind",
                "source_org",
                "notes",
                "source_url",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "source_url": st.column_config.LinkColumn("Source"),
        },
    )

    external_chart = (
        alt.Chart(filtered)
        .mark_bar(cornerRadiusEnd=8)
        .encode(
            x=alt.X("score:Q", title="Published Score", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("display_name:N", title="Model", sort="-x"),
            color=alt.Color("provider:N", title="Provider"),
            column=alt.Column("benchmark:N", title=None),
            tooltip=[
                "display_name",
                "provider",
                "benchmark",
                "benchmark_group",
                "score_display",
                "published_date",
                "source_org",
                "source_kind",
            ],
        )
    )
    st.altair_chart(external_chart, use_container_width=True)


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
