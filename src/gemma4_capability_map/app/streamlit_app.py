from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from gemma4_capability_map.knowledge_work.replay import load_episode_traces
from gemma4_capability_map.metrics.failure_taxonomy import failure_tags
from gemma4_capability_map.traces.replay import load_traces


def main() -> None:
    st.set_page_config(page_title="gemma4-capability-map", layout="wide")
    st.title("gemma4-capability-map")
    st.caption("Failure explorer for reasoning, routing, retrieval, and full-stack Gemma benchmark traces.")
    mode = st.sidebar.selectbox("Explorer mode", ["task_traces", "knowledge_work_episodes"], index=0)
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
