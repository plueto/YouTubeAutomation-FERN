# Chalk Engine + Patch Workspace

This folder is the handoff point for the new chalk-generation engine and its patch.
Keep the engine code, patch, and reference output together here so they can be reviewed
before anything is wired into FERN's live renderer.

## Where to put the patch

- If the patch is a unified Git diff, upload it as `chalk_engine.patch` in this folder.
- If it is a Python patch/installer script, upload it as `engine_patch.py` in this folder.
- If it is a set of source files rather than a patch, put the original files directly in this
  folder and preserve their real filenames. Add a short note here describing the intended entry point.

Do not paste API keys, OAuth tokens, `.env` files, generated videos, or other secrets here.

## Reference output

- `reference_scene_plan.json` is the example scene-plan/quality report supplied for the intended engine.
  It records the target behavior: narration-aligned scene boundaries, multi-page layouts, stroke counts,
  pen timing, camera moves, erase transitions, and synchronized sound effects.

## Integration targets

After the patch is uploaded, review it against the current FERN code before integrating it.
Likely touchpoints include `moduli/chalkboard.py`, `moduli/chalkboard_director.py`,
`tools/preview_chalkboard.py`, and the chalkboard tests.

The engine should remain a distinct, testable module until its dependencies, contracts, timing,
rendering behavior, and failure paths are verified. Do not wire it into the production/upload flow
or publish anything to YouTube during integration testing.

## Expected handoff

1. Inspect and summarize the uploaded patch.
2. Identify dependencies and integration points.
3. Add unit tests or adapt the existing tests.
4. Render and inspect a local preview before considering the integration complete.
