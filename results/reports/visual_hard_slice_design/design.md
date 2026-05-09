# Visual Hard Slice Design

This packet designs the next fresh visual hard slice. It is not model evidence yet.

- generated_at: `2026-05-09T00:39:11.607143+00:00`
- case_count: `8`
- purpose: Design a fresh visual hard slice after v2/v3/v4 catalog-profile results; this is a design packet, not model evidence.

| Case ID | Family | Discriminator | Expected Tool | Argument Focus | Failure Pressure |
| --- | --- | --- | --- | --- | --- |
| visual_form_error_vs_message_author | visual_argument_copying | target_query_region_class_vs_business_subject | extract_layout | target_query should name the visible error or warning region, not message author/source. | v2/v4 tend to select recruiter/note/phone/source concepts instead of executable visual regions. |
| visual_form_error_with_prior_selection_decoy | visual_tool_routing | extract_layout_vs_refine_selection_when_no_real_selection_id | extract_layout | image_id is copied from visual state; target_query stays on visible form error class. | v4 over-preferred refine_selection with selection_id=latest on the form-target case. |
| visual_latest_filter_existing_selection | visual_referent_carryover | compact_filter_query_after_selection_id | refine_selection | selection_id copied exactly; filter_query remains the literal token latest. | v1 expanded latest into latest issue; v2/v4 fixed it. |
| visual_remaining_filter_existing_selection | visual_referent_carryover | compact_filter_query_non_latest_token | refine_selection | filter_query remains remaining without surrounding nouns. | Tests whether the latest-only fix generalizes to other compact selector tokens. |
| visual_region_readback_after_layout_result | visual_region_readback | read_region_text_json_shape | read_region_text | top-level call key remains name and region_id is copied as an opaque id. | v3 emitted tool_name instead of name on readback. |
| visual_metric_panel_vs_table_selector | visual_argument_copying | target_query_specific_visible_region_class | extract_layout | target_query distinguishes metric panel from table without copying business prose. | Tests target_query specificity without relying on validation-error wording. |
| visual_callout_warning_with_user_decoy | visual_argument_copying | target_query_visible_warning_vs_user_decoy | extract_layout | target_query uses warning/callout region even when the user mentions a person or ticket. | Targets the same semantic drift as recruiter note without reusing that surface. |
| visual_selection_id_opaque_copy_with_filter | visual_referent_carryover | opaque_selection_id_copy | refine_selection | selection_id is copied exactly from prior tool result and not replaced with latest/open/etc. | v4 produced selection_id=latest on a case without a valid selection id. |
