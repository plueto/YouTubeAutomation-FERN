"""Generazione sottotitoli SRT dallo script narrato.

Il timing è stimato distribuendo la durata reale dell'audio in modo
proporzionale al numero di parole di ogni blocco — non è una trascrizione
forzata ma per una narrazione TTS a ritmo costante è molto vicina.
"""

import re

MAX_PAROLE_BLOCCO = 12


def _blocchi(script: str) -> list[str]:
    """Spezza lo script in blocchi leggibili: frasi, poi max 12 parole."""
    testo = re.sub(r"\s+", " ", script or "").strip()
    if not testo:
        return []
    frasi = [s.strip() for s in re.split(r"(?<=[.!?])\s+", testo) if s.strip()]
    blocchi = []
    for frase in frasi:
        parole = frase.split()
        for i in range(0, len(parole), MAX_PAROLE_BLOCCO):
            blocchi.append(" ".join(parole[i:i + MAX_PAROLE_BLOCCO]))
    return blocchi


def _timestamp(secondi: float) -> str:
    ms = max(0, int(round(secondi * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def cue_da_script(script: str, durata_audio: float) -> list[dict]:
    """Return the same word-weighted timings as SRT, in memory."""
    blocchi = _blocchi(script)
    if not blocchi or durata_audio <= 0:
        return []
    tot_parole = sum(len(blocco.split()) for blocco in blocchi)
    cues = []
    t = 0.0
    for blocco in blocchi:
        durata = durata_audio * len(blocco.split()) / tot_parole
        fine = min(t + durata, durata_audio)
        cues.append({"start": t, "end": fine, "text": blocco})
        t = fine
    return cues


def genera_srt(script: str, durata_audio: float, output_path: str) -> str | None:
    """Scrive il file SRT e ne ritorna il percorso (None se input vuoto)."""
    cues = cue_da_script(script, durata_audio)
    if not cues:
        return None
    righe = []
    for i, cue in enumerate(cues, 1):
        righe.append(
            f"{i}\n{_timestamp(cue['start'])} --> {_timestamp(cue['end'])}\n"
            f"{cue['text']}\n"
        )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(righe))
    return output_path
