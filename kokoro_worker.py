import argparse
import numpy as np
import soundfile as sf
from kokoro import KPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--voice', default='bm_george')
    args = parser.parse_args()

    with open(args.text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f'[KOKORO] Loading voice: {args.voice}', flush=True)
    pipeline = KPipeline(lang_code='b')

    pieces = []
    for _, _, audio in pipeline(text, voice=args.voice):
        pieces.append(audio)

    if not pieces:
        raise RuntimeError('Kokoro produced no audio')

    audio = np.concatenate(pieces)
    sf.write(args.output, audio, 24000)
    print(f'[KOKORO] Saved: {args.output}', flush=True)


if __name__ == '__main__':
    main()
