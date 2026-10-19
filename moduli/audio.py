import os
import shutil
import subprocess
import sys
import tempfile

from moduli.ffmpeg_utils import ffmpeg_path as _ffmpeg

VOICE_DEFAULT = "bm_george"
# FERN narration is intentionally locked to Kokoro George. No env/preference override.
KOKORO_VOICE = "bm_george"

TTS_TIMEOUT = 45
KOKORO_TIMEOUT = 900
CHUNK_WORDS = 300

_RADICE_PROGETTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_KOKORO_PYTHON = os.path.join(
    _RADICE_PROGETTO,
    "kokoro_env",
    "Scripts",
    "python.exe",
)
_KOKORO_WORKER = os.path.join(
    _RADICE_PROGETTO,
    "kokoro_worker.py",
)


def _split_chunks(text: str, max_words: int = CHUNK_WORDS) -> list[str]:
    sentences = text.replace("\n", " ").split(". ")
    chunks = []
    current = []
    count = 0

    for sentence in sentences:
        words = sentence.split()

        pieces = (
            [
                " ".join(words[i:i + max_words])
                for i in range(0, len(words), max_words)
            ]
            if len(words) > max_words
            else [sentence]
        )

        for piece in pieces:
            word_count = len(piece.split())

            if count + word_count > max_words and current:
                chunks.append(". ".join(current) + ".")
                current = []
                count = 0

            current.append(piece)
            count += word_count

    if current:
        chunks.append(". ".join(current))

    return [c for c in chunks if c.strip(" .")]


def _tts_worker(text: str, output_path: str, voice: str) -> None:
    """Synthesize one chunk. Also writes ``<output>.words.json`` with per-word timestamps
    (seconds, relative to this chunk) when the installed edge-tts can report them."""
    import asyncio
    import json
    import edge_tts

    async def _run():
        try:
            communicate = edge_tts.Communicate(text, voice, boundary="WordBoundary")
        except TypeError:  # older edge-tts: no boundary option, no word timestamps
            await asyncio.wait_for(
                edge_tts.Communicate(text, voice).save(output_path),
                timeout=TTS_TIMEOUT - 5,
            )
            return

        words: list[dict] = []

        async def _stream():
            with open(output_path, "wb") as fh:
                async for part in communicate.stream():
                    if part["type"] == "audio":
                        fh.write(part["data"])
                    elif part["type"] == "WordBoundary":
                        start = part["offset"] / 1e7
                        words.append({"w": part["text"], "s": start, "e": start + part["duration"] / 1e7})

        await asyncio.wait_for(_stream(), timeout=TTS_TIMEOUT - 5)
        if words:
            with open(output_path + ".words.json", "w", encoding="utf-8") as fh:
                json.dump(words, fh)

    asyncio.run(_run())


def _audio_duration(path: str) -> float | None:
    """Duration in seconds via ffprobe (None if unavailable)."""
    probe = shutil.which("ffprobe")
    if not probe:
        return None
    try:
        out = subprocess.run(
            [probe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return float(out)
    except (subprocess.CalledProcessError, ValueError, OSError):
        return None


def _edge_tts_chunk(text: str, output_path: str, voice: str) -> bool:
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                f"import sys; "
                f"sys.path.insert(0, {_RADICE_PROGETTO!r}); "
                f"from moduli.audio import _tts_worker; "
                f"_tts_worker({text!r}, {output_path!r}, {voice!r})"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
    )

    try:
        _, stderr = proc.communicate(timeout=TTS_TIMEOUT)

        if proc.returncode == 0:
            return True

        error = stderr.decode(errors="replace").strip()

        print(
            f"[TTS] Edge TTS errore chunk: {error}",
            flush=True,
        )

        return False

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()

        print(
            f"[TTS] Edge TTS TIMEOUT {TTS_TIMEOUT}s su chunk",
            flush=True,
        )

        return False


def _kokoro_tts(text: str, output_path: str, voice: str) -> bool:
    if not os.path.exists(_KOKORO_PYTHON):
        print(
            f"[KOKORO] Python non trovato: {_KOKORO_PYTHON}",
            flush=True,
        )
        return False

    if not os.path.exists(_KOKORO_WORKER):
        print(
            f"[KOKORO] Worker non trovato: {_KOKORO_WORKER}",
            flush=True,
        )
        return False

    text_file = None
    wav_file = None

    try:
        text_handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            delete=False,
        )
        text_handle.write(text)
        text_handle.close()
        text_file = text_handle.name

        wav_handle = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        )
        wav_handle.close()
        wav_file = wav_handle.name

        print(
            f"[KOKORO] Generazione con voce: {voice}",
            flush=True,
        )

        subprocess.run(
            [
                _KOKORO_PYTHON,
                _KOKORO_WORKER,
                "--text-file",
                text_file,
                "--output",
                wav_file,
                "--voice",
                voice,
            ],
            check=True,
            timeout=KOKORO_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )

        if not os.path.exists(wav_file):
            print(
                "[KOKORO] Nessun file audio prodotto.",
                flush=True,
            )
            return False

        if os.path.getsize(wav_file) < 1024:
            print(
                "[KOKORO] File audio troppo piccolo.",
                flush=True,
            )
            return False

        ff = _ffmpeg()

        print(
            "[KOKORO] Conversione WAV -> MP3...",
            flush=True,
        )

        subprocess.run(
            [
                ff,
                "-y",
                "-nostdin",
                "-v",
                "error",
                "-i",
                wav_file,
                "-ar",
                NORM_SAMPLE_RATE,
                "-ac",
                NORM_CHANNELS,
                "-b:a",
                NORM_BITRATE,
                output_path,
            ],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=600,
        )

        print(
            f"[KOKORO] Audio salvato: {output_path}",
            flush=True,
        )

        return True

    except subprocess.TimeoutExpired:
        print(
            f"[KOKORO] TIMEOUT dopo {KOKORO_TIMEOUT}s.",
            flush=True,
        )
        return False

    except FileNotFoundError as e:
        print(
            f"[KOKORO] File/programa non trovato: {e}",
            flush=True,
        )
        return False

    except subprocess.CalledProcessError as e:
        print(
            f"[KOKORO] Errore processo: {e}",
            flush=True,
        )
        return False

    except Exception as e:
        print(
            f"[KOKORO] Errore: {e}",
            flush=True,
        )
        return False

    finally:
        for path in (text_file, wav_file):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def _concat_audio(parts: list[str], output_path: str) -> None:
    if len(parts) == 1:
        shutil.move(parts[0], output_path)
        return

    list_file = output_path + ".list.txt"

    try:
        ff = _ffmpeg()

        with open(list_file, "w", encoding="utf-8") as f:
            for part in parts:
                f.write(
                    f"file '{os.path.abspath(part)}'\n"
                )

        subprocess.run(
            [
                ff,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_file,
                "-c",
                "copy",
                output_path,
            ],
            check=True,
            capture_output=True,
        )

    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            "Impossibile unire i chunk audio: ffmpeg non disponibile "
            f"o concat fallito. Dettaglio: {e}"
        ) from e

    finally:
        try:
            os.remove(list_file)
        except OSError:
            pass


def _edge_tts(text: str, output_path: str, voice: str) -> bool:
    chunks = _split_chunks(text)

    print(
        f"[TTS] Edge TTS — voce: {voice} — "
        f"{len(chunks)} chunk da ~{CHUNK_WORDS} parole",
        flush=True,
    )

    tmp_parts = []
    all_words: list[dict] = []
    clock = 0.0
    words_complete = True

    try:
        for i, chunk in enumerate(chunks):
            tmp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f"_chunk{i}.mp3",
            )
            tmp = tmp_file.name
            tmp_file.close()

            tmp_parts.append(tmp)

            ok = False

            for attempt in range(2):
                print(
                    f"[TTS] chunk {i + 1}/{len(chunks)} "
                    f"tentativo {attempt + 1}/2...",
                    flush=True,
                )

                if _edge_tts_chunk(chunk, tmp, voice):
                    ok = True
                    break

            if not ok:
                print(
                    f"[TTS] chunk {i + 1} fallito definitivamente",
                    flush=True,
                )
                return False

            # word timestamps for this chunk, shifted by the audio that precedes it
            side = tmp + ".words.json"
            chunk_len = _audio_duration(tmp)
            try:
                with open(side, encoding="utf-8") as fh:
                    import json as _json
                    for w in _json.load(fh):
                        all_words.append({"w": w["w"], "s": clock + w["s"], "e": clock + w["e"]})
            except (OSError, ValueError, KeyError):
                words_complete = False
            if chunk_len is None:
                words_complete = False
            else:
                clock += chunk_len

        _concat_audio(tmp_parts, output_path)
        if words_complete and all_words:
            try:
                from moduli.chalk_voice import write_sidecar
                write_sidecar(output_path, all_words, source=f"edge-tts WordBoundary ({voice})", precision="word")
            except Exception as exc:  # timing is an enhancement; never fail narration over it
                print(f"[TTS] sidecar timestamps non scritti: {exc}", flush=True)
        return True

    finally:
        for part in tmp_parts:
            side = part + ".words.json"
            if os.path.exists(side):
                try:
                    os.remove(side)
                except OSError:
                    pass
            if os.path.exists(part) and part != output_path:
                try:
                    os.remove(part)
                except OSError:
                    pass


def _gtts(text: str, output_path: str, lang: str = "en") -> bool:
    try:
        from gtts import gTTS

        print(
            f"[TTS] Uso gTTS come fallback — lingua: {lang}...",
            flush=True,
        )

        tts = gTTS(
            text=text,
            lang=lang,
            slow=False,
        )

        tts.save(output_path)

        return True

    except Exception as e:
        print(
            f"[TTS] gTTS errore: {e}",
            flush=True,
        )
        return False


def _load_voice() -> str:
    try:
        from moduli.preferenze import carica

        return carica().get(
            "tts_voce",
            VOICE_DEFAULT,
        )

    except Exception:
        return VOICE_DEFAULT


def _kokoro_voice(preferred_voice: str) -> str | None:
    if KOKORO_VOICE:
        return KOKORO_VOICE
    voice = (preferred_voice or "").strip()
    if voice.startswith(("af_", "am_", "bf_", "bm_", "hf_", "hm_")):
        return voice
    return None


def _gtts_lang(voice: str) -> str:
    parts = voice.split("-")
    return parts[0] if parts else "en"


def _audio_valido(output_path: str, parole: int) -> bool:
    try:
        if os.path.getsize(output_path) < 1024:
            return False

    except OSError:
        return False

    try:
        from moduli.montaggio import _media_duration

        durata = _media_duration(output_path)

    except Exception:
        return True

    attesa = parole / 130 * 60

    if attesa > 5 and durata < attesa * 0.4:
        print(
            f"[TTS] Audio sospetto: {durata:.0f}s per "
            f"{parole} parole (attesi ~{attesa:.0f}s)",
            flush=True,
        )
        return False

    return True


LOUDNESS_TARGET_LUFS = os.environ.get(
    "AUDIO_LUFS",
    "-14",
)

NORM_SAMPLE_RATE = os.environ.get(
    "AUDIO_SAMPLE_RATE",
    "48000",
)

NORM_CHANNELS = os.environ.get(
    "AUDIO_CHANNELS",
    "2",
)

NORM_BITRATE = os.environ.get(
    "AUDIO_BITRATE",
    "192k",
)


def _normalizza(path: str) -> None:
    tmp = path + ".norm.mp3"

    try:
        subprocess.run(
            [
                _ffmpeg(),
                "-y",
                "-nostdin",
                "-v",
                "error",
                "-i",
                path,
                "-af",
                (
                    f"loudnorm=I={LOUDNESS_TARGET_LUFS}:"
                    "TP=-1.5:LRA=11"
                ),
                "-ar",
                NORM_SAMPLE_RATE,
                "-ac",
                NORM_CHANNELS,
                "-b:a",
                NORM_BITRATE,
                tmp,
            ],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=600,
        )

        os.replace(tmp, path)

        print(
            f"[TTS] Volume normalizzato a "
            f"{LOUDNESS_TARGET_LUFS} LUFS "
            f"({NORM_SAMPLE_RATE} Hz, "
            f"{NORM_CHANNELS} canali)",
            flush=True,
        )

    except Exception as e:
        print(
            f"[TTS] Normalizzazione saltata ({e}) "
            "— audio originale",
            flush=True,
        )

        try:
            os.remove(tmp)
        except OSError:
            pass


def genera_audio(text: str, output_path: str) -> None:
    """Generate narration with Kokoro bm_george only; never substitute another voice."""
    words = len(text.split())
    voice = "bm_george"
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    print(
        f"[TTS] George-only mode — Kokoro voice: {voice} ({words} parole)...",
        flush=True,
    )
    if not _kokoro_tts(text, output_path, voice):
        raise RuntimeError(
            "George-only TTS failed: Kokoro bm_george could not generate narration. "
            "No alternate voice was used. Fix the Kokoro worker/dependencies and retry."
        )
    if not _audio_valido(output_path, words):
        raise RuntimeError(
            "George-only TTS failed: Kokoro bm_george produced invalid/too-short audio. "
            "No alternate voice was used."
        )
    _normalizza(output_path)
    print(f"[TTS] Audio salvato esclusivamente con Kokoro bm_george: {output_path}", flush=True)
