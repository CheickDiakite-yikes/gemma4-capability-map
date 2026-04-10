# Next Steps

## Immediate

### 1. Build a fully specialist-backed model run for `KnowledgeWorkArena`

Goal:

- move from `hf_service` reasoner plus heuristic router/retriever
- to reasoner + real router + real retriever in a bounded pilot

Why:

- current finished model-backed evidence is strong but not yet fully specialist-backed

### 2. Deepen native artifact grading

Focus:

- workbook cell/value/formula validation
- slide-level section and revision-diff grading
- document field-consistency and section-order validation

Why:

- current grading is structurally strong but still more contract-based than native-office-native

### 3. Expand browser state-machine realism

Focus:

- mid-flow validation failures
- retry/recovery branches
- blocked submissions after partial progress
- approval escalation after artifact creation

Why:

- this is where real agent reliability is tested

## Near-Term

### 4. Run a cross-role finished non-oracle pilot

Target:

- executive assistant
- job application ops
- finance

Why:

- current finished model-backed baseline covers only one executive episode

### 5. Tighten history semantics

Goal:

- make “best by lane” less sensitive to old weighting artifacts
- keep “latest canonical” and “best historical” easy to distinguish

### 6. Add richer defer/escalate/refuse episodes

Focus:

- the correct move after partial progress
- ambiguity and policy traps
- high-cost mutation requests

## Ongoing Discipline

- keep canonical lane pointers clean
- checkpoint long model-backed runs
- append new findings to the research log
- refresh continuity files after every major benchmark pass
