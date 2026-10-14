# Acceptance criteria for the tuned variant (written before it was run)

Variants: A original; B = -6 dB on chalk, body, tick, erase, whoosh (impacts untouched); B2 = -6 dB on every stem
including the reveal impacts (the brief's "transition effects" could include them, so both readings are tested);
C = per-effect tuning chosen from the A/B stem table: chalk -6, body -6, tick -8, erase -6, whoosh -6, thump -3, tap -6.

All are engineering heuristics I chose, not standards. They are proxies; none replaces listening.

* R1  no positive gain anywhere (never boost an effect to make it noticeable)
* R2  full-mix STOI and ESTOI not lower than A, and within 0.003 of the best variant
* R3  speech-to-effects ratio median at least 4 dB better than A
* R4  tick and tap stem peaks at or below -8 dB re the narration RMS; no stem peak above its value in A
* R5  transient prominence of the effects bed at most A + 3 dB
* R6  chalk at least 6 dB above the music bed in the 1.5-7 kHz band while drawing  (if not met: WARNING, do not boost)
* R7  mix sample peak below 0.95 and true peak below -1 dBTP
