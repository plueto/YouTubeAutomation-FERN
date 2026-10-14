# Sound experiment 1 - effect levels (chalk, eraser, whoosh, impacts)

Baseline: commit `6adc4d9` mix (`A_original`). Same program, narration (espeak placeholder voice) and music
(`The Princess - Evening Telecast.mp3`) in every variant; only effect gains differ. Criteria were written before
the tuned variant was run (`CRITERIA.md`). Raw numbers: `metrics.json`. Clips (18 s, 8-26 s of the preview, same
video, only the audio differs): `A_original.mp4`, `B_minus6dB.mp4`, `B2_all_minus6dB.mp4`, `C_tuned.mp4`.

## Measurements (automated proxies)

| metric | A original | B -6 dB (chalk, body, tick, erase, whoosh) | **B2 -6 dB, every effect** | C tuned per effect |
|---|---|---|---|---|
| STOI, full mix | 0.9159 | 0.9244 | **0.9251** | 0.9249 |
| ESTOI, full mix | 0.7566 | 0.7808 | **0.7820** | 0.7816 |
| speech-to-effects, median (dB) | 17.8 | 23.8 | 23.8 | 23.8 |
| speech-to-effects, weakest 10% (dB) | 4.9 | 10.6 | 10.9 | 10.9 |
| effects RMS re voice (dB) | -13.8 | -18.3 | -19.8 | -19.5 |
| tick peak re voice RMS (dB) | -3.1 | -9.1 | -9.1 | -11.1 |
| tap (impact click) peak (dB) | -2.2 | -2.2 | -8.2 | -8.2 |
| thump (impact low) peak (dB) | -3.2 | -3.2 | -9.2 | -6.2 |
| transient prominence of the bed (dB) | 10.4 | **15.2** | 10.4 | 12.5 |
| chalk vs music, 1.5-7 kHz while drawing (dB) | +15.8 | +9.8 | +9.8 | +9.7 |
| mix sum peak before the limiter | 1.004 | 0.985 | 0.985 | 0.985 |
| true peak of the written mix | -2.4 dBTP | -2.4 | -2.4 | -2.4 |
| integrated loudness before normalisation | -21.1 LUFS | -21.2 | -21.2 | -21.2 |

## Criteria (R1-R7) per variant

| | B | B2 | C |
|---|---|---|---|
| R1 no positive gain | PASS | PASS | PASS |
| R2 intelligibility proxies not below A, within 0.003 of best | PASS | PASS | PASS |
| R3 speech-to-effects >= A + 4 dB | PASS (+6.0) | PASS (+6.0) | PASS (+6.0) |
| R4 tick and tap <= -8 dB re voice RMS | **FAIL** (tap -2.2) | PASS | PASS |
| R5 transient prominence <= A + 3 dB | **FAIL** (+4.8) | PASS (+0.0) | PASS (+2.1) |
| R6 chalk >= 6 dB above the music bed | PASS | PASS | PASS |
| R7 true peak <= -1 dBTP | PASS | PASS | PASS |

## Decision
* **B2 adopted** as the new default (every effect stem -6 dB, music unchanged).
* **B rejected**: lowering only chalk and transitions made the untouched impacts relatively *more* prominent
  (+4.8 dB), the opposite of "avoid harsh transients". "Chalk and transition effects" therefore only works as a
  -6 dB change if the reveal impacts move with them.
* **C rejected**: its only difference from B2 is a thump 3 dB louder, whose sole rationale would be making it more
  noticeable; it gains nothing measurable (STOI 0.9249 vs 0.9251) and raises transient prominence by 2.1 dB.
  Independent tuning was evaluated and found not necessary beyond including the impacts.
* Not boosted: no effect is above its original level anywhere.
* -6 dB is an initial experiment, not a final setting: `FERN_CHALK_SFX_DB` (default -6) changes it without code.

## Status of the checks
| check | status |
|---|---|
| the criteria above, for B2 | PASS (all executed and met) |
| defaults reproduce v1 when gains are 0 dB | PASS (golden-reference test) |
| listening to A vs B2 clips | **BLOCKED** - I cannot listen. Whether -6 dB sounds right, and whether the chalk still reads as chalk, needs a human |
| intelligibility with a production voice | **BLOCKED** - measured with the espeak placeholder voice; STOI was built for noise degradation of real speech |
| music bed intelligibility | **WARNING** - narration+music alone scores ESTOI 0.79 (effects-only 0.93); in the weakest 10% of speaking frames the music is level with the speech (0.3 dB) in 300-4000 Hz. Not changed in this pass |
| limiter | info - the mixer scales the whole mix to a fixed peak whenever the sum exceeds 0.97; all variants trigger it (A by a larger margin) |
