# Baseline audit - chalk engine v1 (commit `6adc4d9`, tag `fern-chalk-baseline-1`)

Written **before** any change to product code. Branch for all follow-up work: `tuning/qa-pass-1`; every change
is its own commit so it can be reverted alone.

Status vocabulary (used everywhere in `docs/qa`):

| status | meaning |
|---|---|
| **FAIL** | a *verified* functional, rendering, synchronization or audio defect |
| **WARNING** | executed; a possible issue that needs a human decision |
| **PASS** | the check actually ran and met its criterion |
| **BLOCKED** | could not run (missing dependency, service or input). Never counted as a pass |

Automated measurements (sections 1-4) and artistic judgments (section 5) are kept apart on purpose.

## 1. Baseline and backup verification (change control)

| check | status | evidence |
|---|---|---|
| working tree clean at the baseline commit | PASS | `git status` empty at `6adc4d9` |
| backup created | PASS | `/home/claude/backups/baseline-6adc4d9/`: `repo.bundle`, preview + every input, `SHA256SUMS`, `MANIFEST.md`, `ENVIRONMENT.txt` |
| checksums verify | PASS | `sha256sum -c`: 7/7 OK |
| bundle restores | PASS | cloned into a clean directory, checked out the tag |
| focused tests in the restored clone | PASS | `tests/test_chalk_engine.py`: 41 passed |
| restored clone reproduces the render | PASS | same composed program (530 strokes) and identical frame hash `f465ce4438f235f9` in both checkouts |
| delivered preview equals the repo preview | PASS (with note) | decoded video MD5 `e0cfacb4...` and audio MD5 `14c4f544...` are identical; the *containers* differ by 5.8 KB (re-mux on delivery) |
| full repository suite | BLOCKED (partly) | 187 passed; 16 tests need `googleapiclient` / `telegram` which are not installed here; 2 are scripts that pytest collects as tests (`fixture 'plan' not found`). The same 17 failures + 2 errors exist before and after my changes. They did **not** run, so they are not passes |

## 2. Instruments and their validation

`tools/chalk_artifact_scan.py` is read-only. A PASS from an instrument that cannot fail is meaningless, so it was
validated with a **positive control**: a copy of the baseline video with a deliberate 2-frame brightness pop at
10.0 s and a white bar on the left border from 15-17 s.

* First version missed the pop (it only compared pixels that were dark in both frames, which excludes exactly a
  brightness jump). Fixed to use global luminance statistics.
* Final version detects both injected defects (`frame_pop` at 10.0 s, 101 levels; `content_at_frame_edge` at
  15.0 s) and reports the untouched baseline clean for those two checks (largest step 1.0 level, no border ink).
* Two further instrument bugs were found and fixed by looking at implausible numbers (strokes attributed to the
  wrong scene by a time window; safe-area measured in page rather than screen space).

## 3. Baseline automated results (`docs/qa/scan_baseline/`)

| check | status | detail |
|---|---|---|
| video decodes completely | PASS | 1142 frames, 1280x720 @ 24 fps |
| no ink on the frame border while the camera is still | PASS | |
| no unexplained single-frame luminance jumps | PASS | largest step 1.0 level (threshold 3.0) |
| text large enough | PASS | |
| captions do not overlap drawings | PASS | |
| content keeps 3.5% clear of the frame edge (screen space) | PASS | "DANGEROUS" is 4.2% from the edge in scene 5 |
| no frozen stretch > 2 s | PASS | |
| **captions finish within 1.8 s of their spoken word** | **FAIL** | 6 captions finish 2.77-3.85 s late. Verified by frames, `frames/baseline/sync_evidence_baseline.png`: at t=22.5 s ("dangerous") nothing about danger is on screen; the caption is finished at 25.3 s. Word timing is phrase-interpolated (about +-0.3 s), far smaller than the error |
| finished drawings get >= 0.5 s of dwell | WARNING | scenes 1, 2, 3, 6: 0.10 / 0.29 / 0.38 / 0.13 s |
| captions visible long enough to read | WARNING | CREDIT BUREAU 0.29 s, INVISIBLE SCAFFOLD 0.57 s, NOT SCIENCE FICTION 0.80 s, BEING GRADED 0.82 s |
| accent marks avoid hexagram shapes | WARNING | 8 six-point star accents (see A-001) |
| audio: no NaN, no clipping, no silent gap > 1 s | PASS | peak 0.861, 0 clipped, 0 NaN (earlier analysis of the final mix) |
| audio: chalk sound follows drawing | PASS (proxy) | 3-8 kHz energy 2.5x higher while drawing than idle, correlation 0.54 |
| **listening to the audio** | **BLOCKED** | I cannot listen; loudness judgments below are measurements, not hearing |
| **watching at normal playback speed** | **BLOCKED** | I cannot watch in real time; section 4 is measurement + frame inspection only |

## 4. Motion and pacing measurements (baseline)

| scene | layout | seconds | pen busy | finished dwell (s) | pen speed-up | anchor pull (s) |
|---|---|---|---|---|---|---|
| 1 | statement | 2.09 | 0.54 | 0.10 | 1.3 | 0.0 |
| 2 | chain | 4.47 | 0.59 | 0.29 | 1.5 | 1.8 |
| 3 | bignumber | 5.45 | 0.60 | 0.38 | 1.3 | 0.9 |
| 4 | iceberg | 10.15 | 0.41 | 1.16 | 1.0 | 0.0 |
| 5 | question | 5.13 | 0.37 | 1.84 | 1.0 | 0.0 |
| 6 | statement | 2.19 | 0.55 | 0.13 | 1.6 | 1.8 |
| 7 | network | 9.33 | 0.53 | 0.57 | 1.3 | 0.9 |
| 8 | duo | 8.77 | 0.40 | 2.00 | 1.0 | 0.0 |

* Caption writing speed: median **15.5 letters/s**, range 12.3-22.0 (A-003).
* Camera: single-page pans peak at **133 px/frame** (10% of the frame width), the wrap-around pan at
  **274 px/frame** (21%), without motion blur (A-005).
* Sync offsets, all captions (end of writing minus start of the spoken word): -1.62 ... +3.85 s;
  median about +0.8 s.

## 5. Artistic judgments - illustration quality gate (kept separate from the measurements)

These are my judgments from looking at the baseline frames (`docs/qa/frames/baseline/`). They are not measurements.

### Scene 2 - "Not by a teacher, not by a boss, not by a credit bureau." (2.1-6.6 s, chain)
**Weaknesses.** Three flat icons at one size in a row. "Teacher" is the same generic man-with-a-tie used elsewhere
for "you" and "dangerous". Nothing depicts *not* - the negation that carries the sentence. No hierarchy, overlap
or hatching with intent; the bank is a stock pictogram.
**Proposal - "Three judges, struck out".** Three *distinct, specific* props at different sizes and heights,
asymmetrically staggered so they overlap the frame lines of their panels: a **gradebook** with a pencil (teacher), an
**org chart** whose top box is circled (boss), a **credit-score dial** (300...850 needle) (credit bureau). Hatching
is used only to shade the dial's low zone and the gradebook's cover. After each "not by", a bold yellow
**diagonal strike** is drawn through its prop, once per spoken "not". Captions shrink to one word each.

### Scene 5 - "...how dangerous you might be, and how much effort the model should spend on you." (22.2-27.3 s, question)
**Weaknesses.** A smiling stock person next to a bare "?": the smile contradicts "dangerous", the "?" says nothing
about *effort*, the yellow six-point stars are decoration (and the hexagram problem, A-001). The sentence has two
ideas - danger and rationed effort - and the picture has none of either.
**Proposal - "Two dials".** *Danger*: a large **gauge** (semicircle, graduated ticks) whose needle swings into a
crosshatched yellow-edged red zone. *Effort*: a **tank labelled MODEL** feeding a **pipe and half-closed valve
wheel** that releases only a thin yellow trickle toward a small figure at the end of the pipe. Layered: the tank sits
behind the pipe, the figure in front, the gauge above. The metaphor is *rationing*.

### Scene 8 - "And the score it assigns you? You will never see it. You cannot appeal it. You can barely even prove it exists." (38.8-47.6 s, duo)
**Weaknesses.** The picture shows the score in full (a big "A+" card) while the narration says you *never see it*. An
arrow labelled "CANNOT APPEAL" points at an eye captioned "SEE": captions and picture disagree, and the eye (the
system) is drawn as what the *viewer* sees.
**Proposal - "Behind frosted glass".** A large **frosted pane** (diagonal crosshatch) fills the centre. Behind it, in
dim strokes interrupted by the hatching, a **report card with a "?" for its grade** (occlusion by clipping the
card's lines against the pane). In front, a **hand pressed on the glass** from the left. A watching **eye above**
looks down at the card. After "cannot appeal" a crooked yellow **FINAL stamp** lands on the pane; on "prove it
exists" the card's outline is redrawn as **dashes** (doubtful existence).

All three need new, scene-specific drawings plus clipping/occlusion and dashed strokes; none is "more icons".

## 6. Change plan (each its own commit; revert if it does not visibly help or if anything regresses)

| # | change | revert if |
|---|---|---|
| 1 | sound stems with per-effect gains (defaults bit-identical), then the -6 dB experiment | defaults change; intelligibility proxy drops; transients get harsher |
| 2 | replace six-point accents | any accent reads worse in frames |
| 3 | order the pen by spoken-word time so captions are not late | sync not measurably better, or illustrations become unreadable |
| 4 | text writing speed toward <= 11 letters/s where time allows | more elements dropped than before |
| 5 | transition timing for dwell | next scene's first strokes get cut |
| 6 | motion blur for fast camera moves | ghosting, doubled lines, or slower than 3x render time |
| 7 | the three scene-specific compositions, opt-in via `chalk.art` only | not clearly better in side-by-side frames |

Broad application (the director choosing these automatically) is deliberately **not** part of this pass.
