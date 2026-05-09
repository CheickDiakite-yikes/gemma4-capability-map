# Visual Split Selector Hints Live Replay Decision

Live replay was intentionally skipped for `visual_role_catalog_split_selector_hints_v3`.

Reason:

- Raw exact rate was `1 / 8`, below `visual_role_catalog_argument_hints_v2` at `2 / 8`.
- Executable visual-form recovery stayed `0 / 1`, below `visual_role_catalog_v1` at `1 / 1`.
- The candidate preserved `visual_latest_filter_literal`, but regressed `visual_readback_region_literal` from exact to no-call because the model emitted `tool_name` instead of `name`.

Decision:

- Do not promote v3 to focused visual live replay.
- Keep `visual_role_catalog_argument_hints_v2` as the current best exact visual no-directive candidate.
- Keep `visual_role_catalog_v1` as the executable visual-form routing baseline.

