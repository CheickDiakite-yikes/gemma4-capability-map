import { startTransition, useDeferredValue, useEffect, useRef, useState } from "react";

import {
  fileUrl,
  getHealth,
  getSessionHistory,
  launchSession,
  listApprovals,
  listProfiles,
  listSessions,
  listWorkflows,
  resolveApproval,
  resumeSession,
  retrySession,
  streamSession,
} from "./api";
import type {
  AgentSession,
  ApprovalRequest,
  ProjectGroup,
  SessionHistoryResponse,
  SessionStreamResponse,
  SystemProfile,
  WorkflowCard,
} from "./types";

const ACTIVE_STATUSES = new Set([
  "pending",
  "warming",
  "running",
  "awaiting_approval",
  "resuming",
  "retrying",
]);

const SIGNIFICANT_EVENT_KINDS = new Set([
  "created",
  "instruction_updated",
  "warming",
  "running",
  "resume_requested",
  "resume_started",
  "resumed",
  "approval_required",
  "approval_resolved",
  "artifacts_ready",
  "completed",
  "failed",
  "interrupted",
]);

const NAV_ITEMS = [
  { label: "New chat", accent: "strong" },
  { label: "Search" },
  { label: "Plugins" },
  { label: "Pull requests" },
  { label: "Automations" },
  { label: "Scratchpad" },
];

const DEFAULT_BROWSER_TARGET = "https://www.google.com";

type WorkspaceTab = "summary" | "review" | "browser";
type BackendState = "connecting" | "connected" | "offline";
type TailState = "idle" | "watching" | "error";

type ConversationRow = {
  id: string;
  role: "user" | "agent" | "system";
  label: string;
  body: string;
  meta: string;
  sortKey: string;
};

export function App() {
  const [profiles, setProfiles] = useState<SystemProfile[]>([]);
  const [workflowsByLane, setWorkflowsByLane] = useState<Record<string, WorkflowCard[]>>({
    live_web_stress: [],
    replayable_core: [],
  });
  const [sessions, setSessions] = useState<AgentSession[]>([]);
  const [approvals, setApprovals] = useState<ApprovalRequest[]>([]);
  const [history, setHistory] = useState<SessionHistoryResponse | null>(null);
  const [selectedLane, setSelectedLane] = useState("live_web_stress");
  const [selectedProfileId, setSelectedProfileId] = useState("mlx_gemma4_e2b_reasoner_only");
  const [selectedWorkflowId, setSelectedWorkflowId] = useState("");
  const [selectedProjectId, setSelectedProjectId] = useState("");
  const [selectedSessionId, setSelectedSessionId] = useState("");
  const [projectDraft, setProjectDraft] = useState("gemma-mlx");
  const [instructionDraft, setInstructionDraft] = useState("");
  const [approvalNote, setApprovalNote] = useState("");
  const [search, setSearch] = useState("");
  const [tab, setTab] = useState<WorkspaceTab>("browser");
  const [browserTarget, setBrowserTarget] = useState(DEFAULT_BROWSER_TARGET);
  const [busyAction, setBusyAction] = useState("");
  const [error, setError] = useState("");
  const [streamNotice, setStreamNotice] = useState("");
  const [backendState, setBackendState] = useState<BackendState>("connecting");
  const [tailState, setTailState] = useState<TailState>("idle");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [mobileContextOpen, setMobileContextOpen] = useState(false);
  const [composerExpanded, setComposerExpanded] = useState(false);
  const historyRef = useRef<SessionHistoryResponse | null>(null);

  const deferredSearch = useDeferredValue(search);
  const filteredProjects = filterProjects(groupProjects(sessions, approvals), deferredSearch);
  const selectedSession = sessions.find((session) => session.session_id === selectedSessionId) ?? null;
  const selectedProject = filteredProjects.find((project) => project.projectId === selectedProjectId) ?? null;
  const selectedWorkflow = (workflowsByLane[selectedLane] ?? []).find((workflow) => workflow.workflow_id === selectedWorkflowId) ?? null;
  const selectedProfile = profiles.find((profile) => profile.system_id === selectedProfileId) ?? null;
  const conversationRows = history ? buildConversationRows(history) : [];
  const suggestedWorkflows = (workflowsByLane[selectedLane] ?? []).slice(0, 3);
  const sessionIsActive = Boolean(selectedSession && ACTIVE_STATUSES.has(selectedSession.status));
  const hasConversation = conversationRows.length > 0;

  useEffect(() => {
    historyRef.current = history;
  }, [history]);

  useEffect(() => {
    void bootstrap();
  }, []);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    async function pulse() {
      try {
        await getHealth(controller.signal);
        if (!cancelled) {
          setBackendState("connected");
        }
      } catch (nextError) {
        if (!cancelled && !isAbortError(nextError)) {
          setBackendState("offline");
        }
      }
    }

    void pulse();
    const timer = window.setInterval(() => {
      void pulse();
    }, 12000);

    return () => {
      cancelled = true;
      controller.abort();
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    const laneWorkflows = workflowsByLane[selectedLane] ?? [];
    if (!laneWorkflows.length) {
      return;
    }
    if (!selectedWorkflowId || !laneWorkflows.some((workflow) => workflow.workflow_id === selectedWorkflowId)) {
      setSelectedWorkflowId(laneWorkflows[0].workflow_id);
    }
  }, [selectedLane, selectedWorkflowId, workflowsByLane]);

  useEffect(() => {
    if (profiles.length && !profiles.some((profile) => profile.system_id === selectedProfileId)) {
      const mlxGemma = profiles.find((profile) => profile.system_id === "mlx_gemma4_e2b_reasoner_only");
      setSelectedProfileId((mlxGemma ?? profiles[0]).system_id);
    }
  }, [profiles, selectedProfileId]);

  useEffect(() => {
    if (!filteredProjects.length) {
      setSelectedProjectId("");
      setSelectedSessionId("");
      return;
    }
    const hasProject = filteredProjects.some((project) => project.projectId === selectedProjectId);
    if (!selectedProjectId || !hasProject) {
      setSelectedProjectId(filteredProjects[0].projectId);
      setSelectedSessionId(filteredProjects[0].latest.session_id);
    }
  }, [filteredProjects, selectedProjectId]);

  useEffect(() => {
    if (!selectedProjectId) {
      return;
    }
    const project = filteredProjects.find((item) => item.projectId === selectedProjectId);
    if (!project) {
      return;
    }
    const visible = project.sessions.some((session) => session.session_id === selectedSessionId);
    if (!selectedSessionId || !visible) {
      setSelectedSessionId(project.latest.session_id);
    }
  }, [filteredProjects, selectedProjectId, selectedSessionId]);

  useEffect(() => {
    if (!selectedSessionId) {
      setHistory(null);
      setBrowserTarget(DEFAULT_BROWSER_TARGET);
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const nextHistory = await getSessionHistory(selectedSessionId);
        if (cancelled) {
          return;
        }
        setHistory(nextHistory);
        setBrowserTarget(deriveBrowserTarget(nextHistory));
      } catch (nextError) {
        if (!cancelled) {
          setError(errorMessage(nextError));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedSessionId]);

  useEffect(() => {
    if (!selectedSessionId || !selectedSession || !ACTIVE_STATUSES.has(selectedSession.status)) {
      setTailState("idle");
      return;
    }
    let cancelled = false;
    const controller = new AbortController();

    async function watchSession() {
      setTailState("watching");
      while (!cancelled) {
        try {
          const response = await pullSessionUpdate(selectedSessionId, 10, controller.signal, "auto");
          if (cancelled || !ACTIVE_STATUSES.has(response.session.status)) {
            break;
          }
        } catch (nextError) {
          if (cancelled || isAbortError(nextError)) {
            return;
          }
          setTailState("error");
          setError(errorMessage(nextError));
          return;
        }
      }
      if (!cancelled) {
        setTailState("idle");
      }
    }

    void watchSession();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [selectedSessionId, selectedSession?.status]);

  async function bootstrap() {
    setError("");
    setBackendState("connecting");
    try {
      const [profilesResponse, liveResponse, replayResponse, sessionsResponse, approvalsResponse] = await Promise.all([
        listProfiles(),
        listWorkflows("live_web_stress"),
        listWorkflows("replayable_core"),
        listSessions(),
        listApprovals(),
      ]);
      startTransition(() => {
        setProfiles(profilesResponse.profiles);
        setWorkflowsByLane({
          live_web_stress: liveResponse.workflows,
          replayable_core: replayResponse.workflows,
        });
        setSessions(sessionsResponse.sessions);
        setApprovals(approvalsResponse.approvals);
      });
      setBackendState("connected");
    } catch (nextError) {
      setBackendState("offline");
      setError(errorMessage(nextError));
    }
  }

  async function refreshAfterSession(sessionId: string, projectId?: string) {
    const [sessionsResponse, approvalsResponse, historyResponse] = await Promise.all([
      listSessions(),
      listApprovals(),
      getSessionHistory(sessionId),
    ]);
    startTransition(() => {
      setSessions(mergeSessionCollection(sessionsResponse.sessions, historyResponse.session));
      setApprovals(approvalsResponse.approvals);
      setSelectedSessionId(sessionId);
      if (projectId) {
        setSelectedProjectId(projectId);
      }
      setHistory(historyResponse);
      setBrowserTarget(deriveBrowserTarget(historyResponse));
    });
    setBackendState("connected");
  }

  async function pullSessionUpdate(
    sessionId: string,
    timeoutSeconds: number,
    signal?: AbortSignal,
    mode: "manual" | "auto" = "manual",
  ) {
    const afterSequence = historyRef.current?.events.at(-1)?.sequence ?? 0;
    const [streamResponse, sessionsResponse, approvalsResponse] = await Promise.all([
      streamSession(sessionId, afterSequence, timeoutSeconds, signal),
      listSessions(signal),
      listApprovals(false, signal),
    ]);
    const mergedHistory = mergeHistory(historyRef.current, streamResponse);
    startTransition(() => {
      setSessions(mergeSessionCollection(sessionsResponse.sessions, streamResponse.session));
      setApprovals(approvalsResponse.approvals);
      setHistory(mergedHistory);
      setBrowserTarget(deriveBrowserTarget(mergedHistory));
      setStreamNotice(
        mode === "auto"
          ? streamResponse.events.length
            ? `Watching ${streamResponse.session.title}: ${streamResponse.events.length} new events`
            : `Watching ${streamResponse.session.title}`
          : streamResponse.events.length
            ? `Fetched ${streamResponse.events.length} new events from ${streamResponse.session.title}.`
            : `No new events for ${streamResponse.session.title}.`,
      );
    });
    setBackendState("connected");
    setTailState(mode === "auto" && ACTIVE_STATUSES.has(streamResponse.session.status) ? "watching" : "idle");
    return streamResponse;
  }

  async function handleStartSession() {
    if (!selectedWorkflowId) {
      return;
    }
    setBusyAction("start");
    setError("");
    try {
      const session = await launchSession({
        workflow_id: selectedWorkflowId,
        system_id: selectedProfileId,
        lane: selectedLane,
        project_id: normalizeProjectId(projectDraft),
        human_request: instructionDraft,
        background: true,
      });
      await refreshAfterSession(session.session_id, session.project_id);
      setInstructionDraft("");
      setComposerExpanded(false);
      setStreamNotice(`Started ${session.title} on ${session.system_id}.`);
    } catch (nextError) {
      setError(errorMessage(nextError));
    } finally {
      setBusyAction("");
    }
  }

  async function handleTailSession() {
    if (!selectedSessionId) {
      return;
    }
    try {
      await pullSessionUpdate(selectedSessionId, 2, undefined, "manual");
    } catch (nextError) {
      setError(errorMessage(nextError));
    }
  }

  async function handleResumeSession() {
    if (!selectedSessionId) {
      return;
    }
    setBusyAction("resume");
    setError("");
    try {
      const resumed = await resumeSession(selectedSessionId, instructionDraft, true);
      await refreshAfterSession(resumed.session_id, resumed.project_id);
      setInstructionDraft("");
      setComposerExpanded(false);
      setStreamNotice(`Resumed ${resumed.title}.`);
    } catch (nextError) {
      setError(errorMessage(nextError));
    } finally {
      setBusyAction("");
    }
  }

  async function handleRetrySession() {
    if (!selectedSessionId) {
      return;
    }
    setBusyAction("retry");
    setError("");
    try {
      const retried = await retrySession(selectedSessionId, instructionDraft, true);
      await refreshAfterSession(retried.session_id, retried.project_id);
      setInstructionDraft("");
      setComposerExpanded(false);
      setStreamNotice(`Created retry attempt ${retried.attempt} for ${retried.title}.`);
    } catch (nextError) {
      setError(errorMessage(nextError));
    } finally {
      setBusyAction("");
    }
  }

  async function handleApproval(decision: "approve" | "deny") {
    const approval = selectedSession?.approvals.find((candidate) => candidate.status === "pending");
    if (!approval || !selectedSessionId) {
      return;
    }
    setBusyAction(decision);
    setError("");
    try {
      await resolveApproval(approval.approval_id, decision, approvalNote, decision === "approve");
      await refreshAfterSession(selectedSessionId, selectedProjectId);
      setApprovalNote("");
      setStreamNotice(`Recorded ${decision} for ${approval.title}.`);
    } catch (nextError) {
      setError(errorMessage(nextError));
    } finally {
      setBusyAction("");
    }
  }

  return (
    <div className="app-shell">
      <div className={`workspace-window ${mobileMenuOpen || mobileContextOpen ? "noscroll" : ""}`}>
        <div
          className={`mobile-overlay ${mobileMenuOpen || mobileContextOpen ? "visible" : ""}`}
          onClick={() => {
            setMobileMenuOpen(false);
            setMobileContextOpen(false);
          }}
        />
        <aside className={`sidebar ${mobileMenuOpen ? "mobile-open" : ""}`}>
          <div className="traffic-lights" aria-hidden="true">
            <span className="traffic-light red" />
            <span className="traffic-light amber" />
            <span className="traffic-light green" />
          </div>
          <div className="sidebar-header">
            <div className="sidebar-brand-mark">M</div>
            <div>
              <div className="sidebar-brand">Moonie</div>
              <div className="sidebar-caption">Local-first agent harness</div>
            </div>
          </div>

            <nav className="primary-nav" aria-label="Workspace navigation">
              {NAV_ITEMS.map((item) => (
                <button className={`nav-button ${item.accent ? item.accent : ""}`} key={item.label} type="button">
                  <span className="nav-icon" aria-hidden="true" />
                  <span>{item.label}</span>
                </button>
              ))}
            </nav>

            <div className="sidebar-section">
              <div className="sidebar-section-header">
                <span className="section-label">Projects</span>
                <span className="section-counter">{filteredProjects.length}</span>
              </div>
              <input
                className="project-search"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search projects or threads"
              />
            </div>

            <div className="project-rail">
              {filteredProjects.length ? (
                filteredProjects.map((project) => (
                  <ProjectCard
                    key={project.projectId}
                    onSelectProject={() => {
                      setSelectedProjectId(project.projectId);
                      setSelectedSessionId(project.latest.session_id);
                      setMobileMenuOpen(false);
                    }}
                    onSelectSession={(sessionId) => {
                      setSelectedProjectId(project.projectId);
                      setSelectedSessionId(sessionId);
                      setMobileMenuOpen(false);
                    }}
                    project={project}
                    selectedProjectId={selectedProjectId}
                    selectedSessionId={selectedSessionId}
                  />
                ))
              ) : (
                <EmptyRailCard
                  copy="No matching projects. Start a new local thread and the project rail will organize resumable sessions here."
                  title="No matching projects"
                />
              )}
            </div>

            <div className="sidebar-footer">
              <button className="settings-button" type="button">
                Settings
              </button>
            </div>
          </aside>

          <div className="center-pane-wrapper">
            <header className="topbar-main">
              <div className="topbar-left">
                <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
                  <button className="mobile-toggle" onClick={() => setMobileMenuOpen(true)} type="button">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 12h18M3 6h18M3 18h18"/></svg>
                  </button>
                  <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                    <span className="title-text">New chat</span>
                    <span className="subtitle-text">Gemma 4 on MLX</span>
                  </div>
                </div>
              </div>
              <div className="topbar-actions">
                <span className={`topbar-pill desktop-only ${backendState}`}>API {backendState}</span>
                <span className="topbar-pill desktop-only">Gemma MLX</span>
                <button className="mobile-toggle context-toggle" onClick={() => setMobileContextOpen(true)} type="button">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 20h9M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
                </button>
                <span className={`topbar-pill ${tailState}`}>{tailState === "watching" ? "Live tail" : tailState === "error" ? "Tail error" : "Review"} {approvals.length}</span>
              </div>
            </header>

            <div className="workspace-columns">
              <main className={`conversation-pane ${hasConversation ? "has-history" : "empty-state"}`}>
                <div className={`conversation-header ${hasConversation ? "compact" : ""}`}>
                  <div>
                    <div className="eyebrow">Project workspace</div>
                    <h1 className="conversation-title">{selectedSession?.title ?? "Let’s build"}</h1>
                  </div>

                  <div className="header-meta">
                    <span className="meta-pill">{selectedProfile?.short_label ?? "Loading runtime"}</span>
                    <span className="meta-pill">{humanizeLane(selectedLane)}</span>
                    <span className="meta-pill">{selectedProject?.title ?? "New project"}</span>
                  </div>
                </div>

            {conversationRows.length ? (
              <div className="conversation-feed">
                {conversationRows.map((row) => (
                  <article className={`timeline-card ${row.role}`} key={row.id}>
                    <div className="timeline-heading">
                      <span className="timeline-label">{row.label}</span>
                      <span className="timeline-meta">{row.meta}</span>
                    </div>
                    <div className="timeline-body">{row.body}</div>
                  </article>
                ))}
              </div>
            ) : (
              <div className="hero-panel">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ marginBottom: 16 }}>
                  <path d="M14 4h6v6M10 20H4v-6M20 10v4M4 10v4" />
                </svg>

                <div className="hero-suggestions">
                  {suggestedWorkflows.length ? (
                    suggestedWorkflows.map((workflow) => (
                      <button
                        className="suggestion-card"
                        key={workflow.workflow_id}
                        onClick={() => {
                          setSelectedWorkflowId(workflow.workflow_id);
                          setInstructionDraft(workflow.description);
                        }}
                        type="button"
                      >
                        <div className="suggestion-title">{workflow.title}</div>
                        <div className="suggestion-copy">{workflow.description}</div>
                      </button>
                    ))
                  ) : (
                    <EmptyRailCard
                      copy="Workflow suggestions will appear here once the local API returns the packaged workflow catalog."
                      title="No workflows loaded"
                    />
                  )}
                </div>
              </div>
            )}

            <section className="composer-panel">
              <div className="composer-shell">
                <textarea
                  className="composer-input"
                  value={instructionDraft}
                  onChange={(event) => setInstructionDraft(event.target.value)}
                  placeholder="Ask Gemma to inspect, browse, revise, or continue the selected project with the latest instruction."
                  rows={4}
                />

                <div className="composer-controls" style={{ position: 'relative' }}>
                  <div className="compact-action-row">
                    <div>
                      <input className="compact-input" value={projectDraft} onChange={(e) => setProjectDraft(e.target.value)} placeholder="Project: gemma-mlx" />

                      {profiles.length <= 1 ? (
                        <div className="compact-badge">
                          ⚡ {profiles.length === 1 ? profiles[0].short_label : "Loading..."}
                        </div>
                      ) : (
                        <select className="compact-select" value={selectedProfileId} onChange={(e) => setSelectedProfileId(e.target.value)}>
                          {profiles.map((profile) => (
                            <option key={profile.system_id} value={profile.system_id}>
                              ⚡ {profile.short_label}
                            </option>
                          ))}
                        </select>
                      )}
                    </div>

                    <button
                      className={`icon-btn ${composerExpanded ? "active" : ""}`}
                      onClick={() => setComposerExpanded(!composerExpanded)}
                      type="button"
                      aria-label="Expand settings"
                    >
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
                    </button>
                  </div>

                  {composerExpanded && (
                    <div className="settings-popup">
                      <select className="compact-select" value={selectedLane} onChange={(e) => setSelectedLane(e.target.value)}>
                        {["live_web_stress", "replayable_core"].map((lane) => (
                          <option key={lane} value={lane}>
                            {humanizeLane(lane)}
                          </option>
                        ))}
                      </select>

                      {(workflowsByLane[selectedLane] ?? []).length > 0 && (
                        <select className="compact-select" value={selectedWorkflowId} onChange={(e) => setSelectedWorkflowId(e.target.value)}>
                          <option value="">Workflow...</option>
                          {(workflowsByLane[selectedLane] ?? []).map((workflow) => (
                            <option key={workflow.workflow_id} value={workflow.workflow_id}>
                              {workflow.title}
                            </option>
                          ))}
                        </select>
                      )}
                    </div>
                  )}
                </div>

                <div className="composer-toolbar">
                  <div className="composer-toolbar-left">
                    <button className="toolbar-chip primary" disabled={busyAction !== ""} onClick={handleStartSession} type="button">
                      {busyAction === "start" ? "Starting..." : "Start session"}
                    </button>
                    <button className="toolbar-chip" disabled={!selectedSessionId} onClick={() => void handleTailSession()} type="button">
                      Tail selected
                    </button>
                    <button
                      className="toolbar-chip"
                      disabled={!selectedSession || !selectedSession.resumable || busyAction !== ""}
                      onClick={handleResumeSession}
                      type="button"
                    >
                      {busyAction === "resume" ? "Resuming..." : "Resume"}
                    </button>
                    <button className="toolbar-chip" disabled={!selectedSessionId || busyAction !== ""} onClick={handleRetrySession} type="button">
                      {busyAction === "retry" ? "Retrying..." : "Retry"}
                    </button>
                  </div>

                  <div className="composer-toolbar-right">
                    {selectedWorkflow ? <span className="toolbar-meta">{selectedWorkflow.title}</span> : null}
                  </div>
                </div>
              </div>
            </section>
          </main>

          <section className={`context-pane ${mobileContextOpen ? "mobile-open" : ""}`}>
            <div className="context-pane-header">
              <div className="context-tabs" role="tablist" aria-label="Context tabs">
                {[
                  { id: "summary" as const, label: "Summary" },
                  { id: "review" as const, label: "Review" },
                  { id: "browser" as const, label: "Browser" },
                ].map((item) => (
                  <button
                    className={tab === item.id ? "active" : ""}
                    key={item.id}
                    onClick={() => {
                      setTab(item.id);
                      setMobileContextOpen(false);
                    }}
                    role="tab"
                    type="button"
                  >
                    {item.label}
                  </button>
                ))}
              </div>

              <div className="context-actions">
                <span className={`context-pill ${backendState}`}>{backendState === "connected" ? "API ready" : backendState}</span>
                <span className={`context-pill ${tailState}`}>{sessionIsActive ? "Live session" : selectedSession ? humanizeStatus(selectedSession.status) : "Idle"}</span>
              </div>
            </div>

            <div className="context-pane-body">
              {tab === "summary" ? (
                <SummaryPane
                  approvalNote={approvalNote}
                  busyAction={busyAction}
                  onApprovalNoteChange={setApprovalNote}
                  onApprove={() => void handleApproval("approve")}
                  onDeny={() => void handleApproval("deny")}
                  profile={selectedProfile}
                  session={selectedSession}
                />
              ) : null}

              {tab === "review" ? <ReviewPane history={history} selectedSession={selectedSession} /> : null}

              {tab === "browser" ? (
                <BrowserPane
                  browserTarget={browserTarget}
                  history={history}
                  onBrowserTargetChange={setBrowserTarget}
                  session={selectedSession}
                />
              ) : null}
            </div>
          </section>
        </div>
      </div>

        <footer className="status-strip">
          <div className={`status-item ${backendState}`}>
            <span className="status-label">Backend</span>
            <span>{backendState === "connected" ? "connected" : backendState === "connecting" ? "connecting" : "offline"}</span>
          </div>
          <div className="status-item">
            <span className="status-label">Runtime</span>
            <span>{selectedProfile?.short_label ?? "loading"}</span>
          </div>
          <div className="status-item">
            <span className="status-label">Project</span>
            <span>{selectedProject?.title ?? "new project"}</span>
          </div>
          <div className={`status-item ${tailState}`}>
            <span className="status-label">Session loop</span>
            <span>{tailState === "watching" ? "long-polling" : tailState}</span>
          </div>
          {streamNotice ? (
            <div className="status-item">
              <span className="status-label">Tail</span>
              <span>{streamNotice}</span>
            </div>
          ) : null}
          {error ? (
            <div className="status-item error">
              <span className="status-label">Error</span>
              <span>{error}</span>
            </div>
          ) : null}
        </footer>
      </div>
    </div>
  );
}

function ProjectCard(props: {
  project: ProjectGroup;
  selectedProjectId: string;
  selectedSessionId: string;
  onSelectProject: () => void;
  onSelectSession: (sessionId: string) => void;
}) {
  const { onSelectProject, onSelectSession, project, selectedProjectId, selectedSessionId } = props;
  const selected = project.projectId === selectedProjectId;

  return (
    <article className={`project-card ${selected ? "selected" : ""}`}>
      <button className="project-card-header" onClick={onSelectProject} type="button">
        <div>
          <div className="project-card-title">{project.title}</div>
          <div className="project-card-meta">
            {project.sessions.length} chats · {project.runningCount} running · {project.approvalCount} review
          </div>
        </div>
        <span className="project-state-pill">{selected ? "Open" : "View"}</span>
      </button>

      <div className="project-card-copy">{project.latest.latest_message}</div>

      <div className="thread-list">
        {project.sessions.slice(0, 5).map((session) => (
          <button
            className={`thread-row ${session.session_id === selectedSessionId ? "active" : ""}`}
            key={session.session_id}
            onClick={() => onSelectSession(session.session_id)}
            type="button"
          >
            <div className="thread-row-main">
              <div className="thread-title">{session.title}</div>
              <div className="thread-meta">
                {humanizeWorkflow(session.workflow_category)} · {formatRelative(session.last_activity_at || session.updated_at)}
              </div>
            </div>
            <span className={`mini-status ${session.status}`}>{humanizeStatus(session.status)}</span>
          </button>
        ))}
      </div>
    </article>
  );
}

function SummaryPane(props: {
  session: AgentSession | null;
  profile: SystemProfile | null;
  approvalNote: string;
  onApprovalNoteChange: (value: string) => void;
  onApprove: () => void;
  onDeny: () => void;
  busyAction: string;
}) {
  const { approvalNote, busyAction, onApprovalNoteChange, onApprove, onDeny, profile, session } = props;
  const pendingApproval = session?.approvals.find((approval) => approval.status === "pending") ?? null;
  const metrics = buildMetricCards(session);

  if (!session) {
    return <EmptyContextCard copy="Launch a local session to inspect readiness, repairs, approvals, and runtime posture." title="No active session" />;
  }

  return (
    <div className="context-stack">
      <div className="panel-card">
        <div className="panel-header">
          <div>
            <div className="panel-title">{session.title}</div>
            <div className="panel-copy">
              {humanizeWorkflow(session.workflow_category)} · {humanizeLane(session.lane)} · project {session.project_id}
            </div>
          </div>
          <span className={`state-pill ${session.status}`}>{humanizeStatus(session.status)}</span>
        </div>
      </div>

      <div className="metric-grid">
        {metrics.map((metric) => (
          <div className="metric-card" key={metric.label}>
            <div className="metric-label">{metric.label}</div>
            <div className="metric-value">{metric.value}</div>
            <div className="metric-copy">{metric.description}</div>
          </div>
        ))}
      </div>

      <div className="panel-card">
        <div className="section-label">Runtime posture</div>
        <div className="runtime-grid">
          <div className="runtime-line">
            <span>System</span>
            <strong>{profile?.display_name ?? session.system_id}</strong>
          </div>
          <div className="runtime-line">
            <span>Backend</span>
            <strong>{profile?.backend ?? "unknown"}</strong>
          </div>
          <div className="runtime-line">
            <span>Deployment</span>
            <strong>{profile?.deployment ?? "local"}</strong>
          </div>
          <div className="runtime-line">
            <span>Artifacts</span>
            <strong>{session.artifact_history.length}</strong>
          </div>
        </div>
      </div>

      {pendingApproval ? (
        <div className="panel-card approval-card">
          <div className="section-label">Needs review</div>
          <div className="panel-title small">{pendingApproval.title}</div>
          <div className="panel-copy">{pendingApproval.reason}</div>
          <textarea
            className="approval-textarea"
            value={approvalNote}
            onChange={(event) => onApprovalNoteChange(event.target.value)}
            placeholder="Add an approval note"
            rows={3}
          />
          <div className="approval-actions">
            <button className="action-button primary" disabled={busyAction !== ""} onClick={onApprove} type="button">
              {busyAction === "approve" ? "Approving..." : "Approve"}
            </button>
            <button className="action-button" disabled={busyAction !== ""} onClick={onDeny} type="button">
              {busyAction === "deny" ? "Denying..." : "Deny"}
            </button>
          </div>
        </div>
      ) : null}

      <div className="panel-card">
        <div className="section-label">Tool activity</div>
        {session.tool_invocations.length ? (
          <div className="event-list">
            {session.tool_invocations.slice(-6).map((tool, index) => (
              <div className="event-row" key={`${tool.tool_name}-${index}`}>
                <div>
                  <div className="event-title">{tool.tool_name}</div>
                  <div className="event-copy">{toolArguments(tool.arguments)}</div>
                </div>
                <span className={`state-pill ${tool.validator_result}`}>{tool.validator_result}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="panel-copy">No tool calls recorded yet.</div>
        )}
      </div>
    </div>
  );
}

function ReviewPane(props: { history: SessionHistoryResponse | null; selectedSession: AgentSession | null }) {
  const { history, selectedSession } = props;
  if (!history || !selectedSession) {
    return <EmptyContextCard copy="Artifacts, revision history, and review notes appear here once a thread produces them." title="No review data yet" />;
  }

  const latestArtifact = selectedSession.latest_artifact_path;
  const latestPath = latestArtifact ? fileUrl(latestArtifact) : "";

  return (
    <div className="context-stack">
      <div className="panel-card">
        <div className="section-label">Artifact history</div>
        {history.artifact_history.length ? (
          <div className="event-list">
            {history.artifact_history
              .slice()
              .reverse()
              .slice(0, 5)
              .map((artifact) => (
                <div className="event-row stacked" key={artifact.artifact_revision_id}>
                  <div className="event-title">{artifact.title || artifact.artifact_id}</div>
                  <div className="event-copy">Revision {artifact.revision} · {formatRelative(artifact.created_at)}</div>
                  <div className="event-path">{artifact.file_path}</div>
                </div>
              ))}
          </div>
        ) : (
          <div className="panel-copy">No artifact revisions recorded yet.</div>
        )}
      </div>

      {latestPath && isRenderableImage(selectedSession.latest_artifact_path) ? (
        <div className="panel-card">
          <div className="section-label">Latest preview</div>
          <img alt="Latest artifact preview" className="artifact-preview" src={latestPath} />
        </div>
      ) : null}

      <div className="panel-card">
        <div className="section-label">Instruction history</div>
        {history.instruction_history.length ? (
          <div className="instruction-stack">
            {history.instruction_history
              .slice()
              .reverse()
              .slice(0, 4)
              .map((instruction) => (
                <div className="instruction-row" key={instruction.instruction_id}>
                  <div className="event-copy strong">{instruction.content}</div>
                  <div className="event-copy">{instruction.source} · {formatRelative(instruction.created_at)}</div>
                </div>
              ))}
          </div>
        ) : (
          <div className="panel-copy">No instruction history captured yet.</div>
        )}
      </div>

      <div className="panel-card">
        <div className="section-label">Latest review note</div>
        <div className="panel-copy">{selectedSession.latest_review_feedback || "No explicit review feedback yet."}</div>
      </div>
    </div>
  );
}

function BrowserPane(props: {
  browserTarget: string;
  onBrowserTargetChange: (value: string) => void;
  history: SessionHistoryResponse | null;
  session: AgentSession | null;
}) {
  const { browserTarget, history, onBrowserTargetChange, session } = props;
  const browserEvents = (history?.events ?? []).filter((event) => event.kind === "tool_call_result");
  const previewSource = browserPreviewSource(session, browserTarget);

  return (
    <div className="context-stack browser-stack">
      <div className="browser-shell">
        <div className="browser-toolbar">
          <div className="browser-nav">
            <button className="browser-button" type="button">
              &lt;
            </button>
            <button className="browser-button" type="button">
              &gt;
            </button>
            <button className="browser-button" type="button">
              R
            </button>
          </div>
          <input className="browser-input" onChange={(event) => onBrowserTargetChange(event.target.value)} value={browserTarget} />
        </div>

        <div className="browser-canvas">
          {previewSource ? (
            <img alt="Browser preview" className="browser-image" src={previewSource} />
          ) : isEmbedFriendly(browserTarget) ? (
            <iframe className="browser-frame" src={browserTarget} title="Browser preview" />
          ) : (
            <div className="browser-fallback">
              <div className="browser-fallback-title">Desktop webview target</div>
              <div className="browser-fallback-copy">
                The reference UI implies a privileged browser pane. In the web build, Moonie keeps the same shell and browser state model, but external sites like Google still need an Electron or native webview host to behave like the desktop reference.
              </div>
              <a className="browser-link" href={browserTarget} rel="noreferrer" target="_blank">
                Open target externally
              </a>
            </div>
          )}
        </div>
      </div>

      <div className="panel-card">
        <div className="section-label">Browser state</div>
        {browserEvents.length ? (
          <div className="event-list">
            {browserEvents.slice(-6).map((event) => (
              <div className="event-row" key={event.event_id}>
                <div>
                  <div className="event-title">{event.message}</div>
                  <div className="event-copy">{formatRelative(event.created_at)}</div>
                </div>
                <span className="state-pill completed">event</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="panel-copy">No browser-side events captured yet.</div>
        )}
      </div>
    </div>
  );
}

function EmptyRailCard(props: { title: string; copy: string }) {
  return (
    <div className="empty-card">
      <div className="empty-title">{props.title}</div>
      <div className="empty-copy">{props.copy}</div>
    </div>
  );
}

function EmptyContextCard(props: { title: string; copy: string }) {
  return (
    <div className="panel-card empty-panel">
      <div className="panel-title">{props.title}</div>
      <div className="panel-copy">{props.copy}</div>
    </div>
  );
}

function groupProjects(sessions: AgentSession[], approvals: ApprovalRequest[]): ProjectGroup[] {
  const grouped = new Map<string, AgentSession[]>();
  const approvalSessionIds = new Set(approvals.map((approval) => approval.session_id));

  for (const session of sessions) {
    const key = session.project_id || "general";
    const existing = grouped.get(key) ?? [];
    existing.push(session);
    grouped.set(key, existing);
  }

  const groups: ProjectGroup[] = [];
  for (const [projectId, projectSessions] of grouped.entries()) {
    const orderedSessions = [...projectSessions].sort((left, right) =>
      (right.last_activity_at || right.updated_at).localeCompare(left.last_activity_at || left.updated_at),
    );
    const latest = orderedSessions[0];
    groups.push({
      projectId,
      title: humanizeProjectId(projectId),
      sessions: orderedSessions,
      latest,
      approvalCount: orderedSessions.filter((session) => approvalSessionIds.has(session.session_id)).length,
      runningCount: orderedSessions.filter((session) => ACTIVE_STATUSES.has(session.status)).length,
    });
  }

  return groups.sort((left, right) => {
    const leftScore = left.approvalCount * 10 + left.runningCount;
    const rightScore = right.approvalCount * 10 + right.runningCount;
    if (rightScore !== leftScore) {
      return rightScore - leftScore;
    }
    return (right.latest.last_activity_at || right.latest.updated_at).localeCompare(left.latest.last_activity_at || left.latest.updated_at);
  });
}

function filterProjects(projects: ProjectGroup[], search: string) {
  if (!search.trim()) {
    return projects;
  }
  const lowered = search.trim().toLowerCase();
  return projects.filter((project) => {
    if (project.title.toLowerCase().includes(lowered) || project.projectId.toLowerCase().includes(lowered)) {
      return true;
    }
    return project.sessions.some((session) => session.title.toLowerCase().includes(lowered));
  });
}

function buildConversationRows(history: SessionHistoryResponse): ConversationRow[] {
  const rows: ConversationRow[] = [];

  for (const instruction of history.instruction_history) {
    rows.push({
      id: instruction.instruction_id,
      role: "user",
      label: "You",
      body: instruction.content,
      meta: `${instruction.source} · ${formatRelative(instruction.created_at)}`,
      sortKey: instruction.created_at,
    });
  }

  for (const event of history.events) {
    if (!SIGNIFICANT_EVENT_KINDS.has(event.kind)) {
      continue;
    }
    rows.push({
      id: event.event_id,
      role: event.kind.includes("approval") || event.kind === "failed" || event.kind === "interrupted" ? "system" : "agent",
      label: humanizeStatus(event.kind),
      body: event.message,
      meta: `#${event.sequence} · ${formatRelative(event.created_at)}`,
      sortKey: `${event.created_at}:${String(event.sequence).padStart(4, "0")}`,
    });
  }

  return rows.sort((left, right) => left.sortKey.localeCompare(right.sortKey)).slice(-18);
}

function buildMetricCards(session: AgentSession | null) {
  if (!session) {
    return [];
  }

  return [
    metricCard("Readiness", session.metrics.role_readiness_score, "How usable the final work is."),
    metricCard("Artifact", session.metrics.artifact_quality_score, "Quality of the produced document or asset."),
    metricCard("Browser", session.metrics.browser_workflow_score, "How cleanly the browser workflow landed."),
    metricCard("Strict", session.metrics.strict_interface_score, "How well the run obeyed the contract."),
    metricCard("Recovered", session.metrics.recovered_execution_score, "Whether it still recovered and finished."),
    metricCard("Plan clean", session.metrics.raw_planning_clean_rate, "How often the plan worked without repairs."),
    metricCard("Repairs", session.metrics.controller_repair_count, "How often the controller had to step in.", true),
  ].filter((item): item is { label: string; value: string; description: string } => item !== null);
}

function metricCard(
  label: string,
  value: string | number | boolean | undefined,
  description: string,
  rawCount = false,
) {
  if (typeof value === "undefined") {
    return null;
  }
  if (rawCount) {
    return { label, value: `${Number(value)}`, description };
  }
  return { label, value: `${(Number(value) * 100).toFixed(1)}%`, description };
}

function mergeHistory(previous: SessionHistoryResponse | null, streamResponse: SessionStreamResponse): SessionHistoryResponse {
  if (!previous || previous.session.session_id !== streamResponse.session.session_id) {
    return {
      session: streamResponse.session,
      instruction_history: streamResponse.session.instruction_history,
      artifact_history: streamResponse.session.artifact_history,
      events: streamResponse.events,
      runtime_trace: streamResponse.session.runtime_trace ?? undefined,
    };
  }

  const mergedEvents = [...previous.events];
  for (const event of streamResponse.events) {
    if (!mergedEvents.some((candidate) => candidate.event_id === event.event_id)) {
      mergedEvents.push(event);
    }
  }

  return {
    session: streamResponse.session,
    instruction_history: streamResponse.session.instruction_history,
    artifact_history: streamResponse.session.artifact_history,
    events: mergedEvents.sort((left, right) => left.sequence - right.sequence),
    runtime_trace: streamResponse.session.runtime_trace ?? undefined,
  };
}

function mergeSessionCollection(sessions: AgentSession[], session: AgentSession) {
  const seen = new Set<string>();
  const merged: AgentSession[] = [session];
  seen.add(session.session_id);

  for (const candidate of sessions) {
    if (seen.has(candidate.session_id)) {
      continue;
    }
    merged.push(candidate);
    seen.add(candidate.session_id);
  }

  return merged;
}

function deriveBrowserTarget(history: SessionHistoryResponse | null) {
  const previewAsset = history?.session.preview_asset;
  if (previewAsset && isRenderableImage(previewAsset)) {
    return fileUrl(previewAsset);
  }
  const latestArtifact = history?.session.latest_artifact_path;
  if (latestArtifact && isRenderableImage(latestArtifact)) {
    return fileUrl(latestArtifact);
  }
  return DEFAULT_BROWSER_TARGET;
}

function browserPreviewSource(session: AgentSession | null, browserTarget: string) {
  if (browserTarget.includes("/v1/files?path=")) {
    return browserTarget;
  }
  if (session?.latest_artifact_path && isRenderableImage(session.latest_artifact_path)) {
    return fileUrl(session.latest_artifact_path);
  }
  if (session?.preview_asset && isRenderableImage(session.preview_asset)) {
    return fileUrl(session.preview_asset);
  }
  return "";
}

function normalizeProjectId(value: string) {
  const normalized = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
  return normalized || "gemma-mlx";
}

function humanizeProjectId(value: string) {
  return value
    .replace(/[_-]+/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((part) => part[0]?.toUpperCase() + part.slice(1))
    .join(" ");
}

function humanizeStatus(value: string) {
  return value.replace(/_/g, " ");
}

function humanizeWorkflow(value: string) {
  return value.replace(/_/g, " ");
}

function humanizeLane(value: string) {
  return value.replace(/_/g, " ");
}

function toolArguments(argumentsValue: Record<string, unknown>) {
  return Object.entries(argumentsValue)
    .slice(0, 3)
    .map(([key, value]) => `${key}=${String(value)}`)
    .join(", ");
}

function formatRelative(value: string) {
  if (!value) {
    return "now";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function isRenderableImage(path: string | null | undefined) {
  if (!path) {
    return false;
  }
  return /\.(png|jpg|jpeg|webp)$/i.test(path);
}

function isEmbedFriendly(target: string) {
  try {
    const url = new URL(target);
    return !/google\.com$/i.test(url.hostname);
  } catch {
    return false;
  }
}

function errorMessage(value: unknown) {
  if (value instanceof Error) {
    return value.message;
  }
  return String(value);
}

function isAbortError(value: unknown) {
  return value instanceof Error && value.name === "AbortError";
}
