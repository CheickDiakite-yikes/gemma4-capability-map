# Failure Taxonomy

- `failed`: top-level task failure marker.
- `answer_missing`: no final answer was produced.
- `answer_mismatch`: a final answer was produced but did not satisfy the task scorer.
- `generation_truncated`: the decode budget was exhausted before the answer completed.
- `thinking_overflow`: thought text consumed the decode budget and the final answer never arrived.
- `image_grounding_miss`: the model saw a screenshot/doc image but answered with the wrong grounded setting/action.
- `arg_mismatch`: the selected tool is correct but one or more arguments are wrong or incomplete.
- `malformed_call`: the raw tool request could not be parsed into the canonical AST.
- `hallucinated_tool`: the model invoked a tool that was not present in the declared tool set.
- `wrong_tool`: the tool choice scorer marked the selected tool as incorrect.
- `wrong_final_state`: tool execution completed but the resulting state was wrong.
- `retrieval_miss`: retrieval recall/evidence scoring failed.

Some of these are direct evaluator outcomes and some are derived from trace inspection. The goal is to separate benchmark-harness issues from true capability failures.
