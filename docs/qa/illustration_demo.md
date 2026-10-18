# Opt-in hand-composed illustration demo

The three hand-composed drawings in `moduli/chalk_scene_art.py` remain opt-in. The default director and ordinary preview do not select them automatically.

## Render the demonstration

Use the review branch, an available Python environment with the project dependencies, and local FFmpeg/FFprobe. Point the preview at narration that includes these concepts: teacher/boss/credit bureau; dangerous/effort; and a hidden score that is assigned, never seen, cannot be appealed, and is hard to prove.

Example on Windows PowerShell:

```powershell
$project = "$HOME\Desktop\YouTubeAutomation_FERN"
$review = "$HOME\Desktop\FERN_QA_PATCH_REVIEW"
$py = "$HOME\AppData\Local\Programs\Python\Python313\python.exe"

& $py "$review\tools\preview_chalkboard.py" `
  --duration 75 `
  --audio "$project\output\narration.mp3" `
  --script "$project\director_input\script.txt" `
  --output "$review\output\fern_chalk_illustration_demo.mp4" `
  --illustration-demo --strict-quality
```

Choose an output filename that does not already exist. The preview refuses to overwrite an existing MP4 unless `--force` is explicitly supplied.

The demo planner searches for contiguous narration windows containing **all** anchor phrases for each composition. It may combine up to three adjacent beats to preserve a single coherent spoken idea and enough drawing time. If the narration excerpt does not contain every anchor, or if the three windows overlap, the tool stops with a diagnostic instead of drawing unrelated art on a generic scene. Increase `--duration` to include the complete phrases; do not weaken the anchor requirement simply to make the command finish.

## QA status meanings

- `PASS`: the check ran and met its criterion.
- `WARNING`: the check ran and found a non-fatal issue.
- `FAIL`: the check ran and detected a failed criterion.
- `BLOCKED`: an input or measurement needed for the check was unavailable. This is never reported as a pass.

Ordinary QA returns a nonzero exit code for a `FAIL`. With `--strict-quality`, a `BLOCKED` check also returns a nonzero code. Warnings remain visible in `qa_report.json` and the console output.

The automated demo is intended to prove that the three art builders can be selected, scheduled against real narration phrases, composed, rendered, and checked. It is not a substitute for watching the video at normal speed; camera rhythm, drawing legibility, and whether the illustration reads intuitively still require human review. Test output is local and must never call the YouTube publisher.
