# Visual Schema Field Hints Live Replay Decision

Live replay was intentionally skipped for `visual_role_catalog_schema_field_hints_v4`.

Reason:

- Raw exact rate was `2 / 8`, tying `visual_role_catalog_argument_hints_v2` but not improving it.
- Executable visual-form recovery stayed `0 / 1`, below `visual_role_catalog_v1` at `1 / 1`.
- The candidate preserved exact `visual_latest_filter_literal` and exact `visual_readback_region_literal`.
- The remaining form-target case regressed to wrong visual-tool routing: `refine_selection(selection_id="latest", filter_query="phone issue")` instead of an executable `extract_layout` call.

Decision:

- Do not promote v4 to focused visual live replay.
- Keep `visual_role_catalog_argument_hints_v2` as the current best exact visual no-directive candidate.
- Keep `visual_role_catalog_v1` as the executable visual-form routing baseline.
- Next test should use a fresh visual hard slice or a narrower field-level mechanism that does not over-prefer `refine_selection`.

