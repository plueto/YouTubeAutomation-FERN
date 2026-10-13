# FERN chalk engine

The chalkboard mode renders narrated, story-driven chalk documentaries.  This is how it works and where to
change things.

```
narration.mp3 (+ narration.mp3.words.json)         script text
          \                                          /
           chalkboard_director.build_scene_plan   -> scenes with start/end, word_times and a `chalk` plan
                          |                              (chalk_story: layout, illustrations, captions, anchors)
           chalk_composer.compose_program         -> timed strokes, camera keys, erases, sound cues
                          |
           chalk_engine.ChalkEngine.render(t)     -> frames (persistent chalk layer, camera, eraser, dust)
                          |
           chalk_sound.build_soundtrack           -> narration + chalk/eraser/impact/whoosh + ducked music
                          |
                       ffmpeg  ->  video.mp4  (+ .scene_plan.json, .chalk_report.json)
```

## Modules
| module | role |
|---|---|
| `chalk_engine.py` | board texture, grainy chalk, incremental stroke painting, camera, eraser + ghost smudges, dust |
| `chalk_lettering.py` | single-line stroke font; text is *written* in stroke order with hand jitter |
| `chalk_illustrations.py` | 57 procedural illustrations in 4 detail stages; SVG import; keyword lexicon |
| `chalk_composer.py` | 12 layouts; word anchoring; pen scheduling; pacing decisions; camera + transitions |
| `chalk_story.py` | local director: beats, layout choice, captions, anchors; validation of AI output |
| `chalk_voice.py` | word-timing sidecars, TTS->script alignment, local espeak-ng placeholder voice |
| `chalk_sound.py` | procedural sound design and the music mix |
| `chalk_render.py` | end-to-end entry point used by `render_chalkboard_video` |

## Running it
```
python tools/make_chalk_preview.py                    # first paragraph of director_input/script.txt
python tools/chalk_qa.py output/preview/fern_chalk_preview.mp4
```
`FERN_CHALK_ENGINE=legacy` selects the old Pillow renderer; `FERN_CHALK_BEAT_SECONDS` (default 7.5) sets the
target beat length.

## Adding artwork
```python
from moduli.chalk_illustrations import register_svg
register_svg("dossier", ["M0 0 L100 0 L100 140 L0 140 Z",              # outline (stage 0)
                         ("M15 30 L85 30 M15 50 L85 50", 1, "dim"),     # detail  (stage 1)
                         ("M20 100 L80 100", 3, "yellow")],             # accent  (stage 3)
             keywords=("dossier", "file on"))
```
Or write a function returning `Part`s in the unit square and add it to `LIBRARY` / `KEYWORDS`.

## Timing honesty
`plan["timing_source"]` says how the drawings are synced: `tts_word_boundaries` (Edge TTS, word accurate),
`tts_phrase_boundaries_interpolated` (phrase accurate), `local_asr_word_timestamps`, or an estimate.  The
composer also records per scene the pacing decisions it had to take (`.chalk_report.json` -> `pacing`).

## Known limits
* No drawing hand: a convincing one was judged not achievable procedurally; dust marks the chalk tip instead.
* Words are never "rewritten" (scribble-out and correct) - only erased with the eraser.
* The local illustrations are icon-grade; very specific story objects still need `register_svg` artwork.
