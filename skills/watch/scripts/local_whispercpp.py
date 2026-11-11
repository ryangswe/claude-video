"""Subprocess-only whisper.cpp adapter: GPU (Metal) on Apple Silicon, CPU elsewhere.

whisper.cpp runs Silero voice activity detection first, so long silences and music do
not turn into invented text. It also starts each 30-second window without the text before
it (-mc 0). With that text, a long talk fell into loops that repeated one sentence dozens
of times, at different places on different runs.

It reports one entry per word; the words are regrouped into sentence-sized segments
here. Around a silence, whisper.cpp stretches one word
across the gap, so an entry much longer than its text needs to be spoken marks a
boundary. In scripts written without spaces (Chinese, Japanese, Thai...) an entry is a
whole phrase, so the allowance grows with the text and uses a slower per-character rate.
The script is read from each entry, because the detected language can be wrong for part
of a mixed-language recording.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

from config import ConfigError, get_config, positive_number
from local_whisperx import run_inference
from runtime import diagnostic
from transcribe import Segments, normalize_segments

INSTALL = '--install-whispercpp'
_HF = 'https://huggingface.co'
# name -> (file, url, sha256, bytes). Checksums match Hugging Face's published LFS hashes.
MODELS = {
    'large-v3-turbo': ('ggml-large-v3-turbo.bin', f'{_HF}/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin',
                       '1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69', 1624555275),
    'small': ('ggml-small.bin', f'{_HF}/ggerganov/whisper.cpp/resolve/main/ggml-small.bin',
              '1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b', 487601967),
}
VAD_MODEL = ('ggml-silero-v5.1.2.bin', f'{_HF}/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin',
             '29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf', 885098)

GAP_WORD = 2.0      # seconds an entry may run beyond its spoken length before it counts as stretched
PAUSE = 1.0         # seconds between words that start a new segment
MIN_SENTENCE = 40   # characters before a sentence end closes a segment
MAX_SEGMENT = 200   # characters before a segment is closed anyway
CHAR_SECONDS = 0.12             # generous speaking time per character, alphabetic scripts
CHAR_SECONDS_NO_SPACE = 0.35    # generous, for spotting a stretched phrase in a no-space script
PHRASE_CHAR_SECONDS = 0.2       # typical, for placing the start of a stretched no-space phrase
# Thai, Lao, Tibetan, Myanmar, Khmer, kana, CJK ideographs.
NO_SPACE_SCRIPT = re.compile('[\u0e00-\u0eff\u0f00-\u0fff\u1000-\u109f\u1780-\u17ff'
                             '\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')
ENDS_SENTENCE = re.compile(r'[.?!,;:。？！，；：、]["\')\]」』]*$')   # the word closes a clause
SENTENCE_END = re.compile(r'[.?!。？！]["\')\]」』]*$')


def settings(cfg=None) -> dict:
    cfg = dict(get_config(backend_override='whispercpp') if cfg is None else cfg)
    if cfg['whispercpp_model'] not in MODELS:
        raise ConfigError(f"WATCH_WHISPERCPP_MODEL must be one of: {', '.join(sorted(MODELS))}.")
    cfg['whispercpp_timeout'] = (positive_number(cfg['whispercpp_timeout'], 'WATCH_WHISPERCPP_TIMEOUT')
                                 if cfg['whispercpp_timeout'] else None)
    if cfg['whispercpp_language'] and not re.fullmatch('[a-z]{2,3}', cfg['whispercpp_language']):
        raise ConfigError('WATCH_WHISPERCPP_LANGUAGE must be a two- or three-letter lowercase Whisper code.')
    return cfg


def model_dir() -> Path:
    return Path.home() / '.cache' / 'watch' / 'whispercpp'


def model_path(name: str) -> Path:
    return model_dir() / MODELS[name][0]


def vad_path() -> Path:
    return model_dir() / VAD_MODEL[0]


def sentinel() -> Path:
    return model_dir() / '.deps-ok'


def executable_path(cfg: dict, *, require_ready=True) -> Path:
    value = cfg['whispercpp_bin']
    if not value or not Path(value).is_absolute() or not Path(value).is_file():
        raise SystemExit(f'whisper.cpp (whisper-cli) missing. Run setup.py {INSTALL}.')
    if require_ready and not (sentinel().is_file() and model_path(cfg['whispercpp_model']).is_file()
                              and vad_path().is_file()):
        raise SystemExit(f'whisper.cpp models are incomplete. Run setup.py {INSTALL} to download them.')
    return Path(value)


def command(executable: Path, audio: Path, output_base: Path, cfg: dict) -> list[str]:
    threads = str(max(1, min(8, os.cpu_count() or 4)))
    return [str(executable), '-m', str(model_path(cfg['whispercpp_model'])), '-f', str(audio.resolve()),
            '-oj', '-of', str(output_base), '-l', cfg['whispercpp_language'] or 'auto', '-t', threads, '-np',
            '--vad', '-vm', str(vad_path()), '-ml', '1', '-sow', '-mc', '0']


def regroup(words: list[tuple[float, float, str]]) -> list[dict]:
    """Join per-word entries into segments; see the module docstring."""
    segments, current = [], []
    def flush():
        text = ''.join(word[2] for word in current).strip()
        if text:
            start = max(current[0][0], segments[-1]['start'] if segments else 0.0)
            segments.append({'start': start, 'end': max(start, current[-1][1]), 'text': text})
        current.clear()
    for start, end, text in words:
        word = text.strip()
        if not word:
            continue
        phrase = bool(NO_SPACE_SCRIPT.search(word))
        spoken = 0.2 + (CHAR_SECONDS_NO_SPACE if phrase else CHAR_SECONDS) * len(word)
        last_end = current[-1][1] if current else (segments[-1]['end'] if segments else None)
        after_pause = last_end is not None and start - last_end > PAUSE
        if after_pause:
            flush()
        if end - start > spoken + GAP_WORD:
            closes = ENDS_SENTENCE.search(word) or not re.search(r'\w', word)
            # In every real run, whisper.cpp stretched a phrase back over the silence before it
            # and kept its end accurate.
            if closes and not phrase and not after_pause and (current or segments):
                # It ends the speech before the silence.
                end = start + spoken
                if current:
                    current.append((start, end, text))
                    flush()
                else:
                    segments[-1]['text'] = (segments[-1]['text'] + text).strip()
                    segments[-1]['end'] = max(segments[-1]['end'], end)
                continue
            # It opens the speech after the silence.
            flush()
            start = end - (0.2 + PHRASE_CHAR_SECONDS * len(word) if phrase else spoken)
        if current and len(''.join(item[2] for item in current) + text) > MAX_SEGMENT:
            flush()
        current.append((start, end, text))
        if SENTENCE_END.search(word) and len(''.join(item[2] for item in current)) >= MIN_SENTENCE:
            flush()
    flush()
    return segments


def parse_output(path: Path, language_hint: str = '') -> Segments:
    data = json.loads(path.read_bytes().decode('utf-8', errors='replace'))
    language = language_hint or (data.get('result') or {}).get('language')
    words = [(entry['offsets']['from'] / 1000, entry['offsets']['to'] / 1000, entry['text'])
             for entry in data['transcription']]
    segments = normalize_segments({'segments': regroup(words)})
    segments.language = language
    return segments


def transcribe_audio(audio: Path, cfg=None, *, require_ready=True):
    cfg = settings(cfg)
    executable = executable_path(cfg, require_ready=require_ready)
    print(f"[watch] local whisper.cpp {cfg['whispercpp_model']} transcription with voice activity detection.",
          file=sys.stderr, flush=True)
    with tempfile.TemporaryDirectory(prefix='watch-whispercpp-', dir=audio.parent) as temporary:
        base = Path(temporary) / audio.stem
        tail = run_inference(command(executable, audio, base, cfg), cfg['whispercpp_timeout'], name='whisper.cpp',
                             install=INSTALL, timeout_var='WATCH_WHISPERCPP_TIMEOUT', echo=False,
                             memory_hint='Memory or device failure; try WATCH_WHISPERCPP_MODEL=small.')
        try:
            segments = parse_output(Path(f'{base}.json'), cfg['whispercpp_language'])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            # whisper-cli exits 0 on a bad option and prints its usage after the error line.
            errors = [line.strip() for line in tail.decode('utf-8', errors='replace').splitlines()
                      if line.strip().lower().startswith('error')]
            raise SystemExit(f'whisper.cpp output missing or malformed ({type(exc).__name__}).\n'
                             + ''.join(f'{line}\n' for line in errors[:5]) + diagnostic(tail)) from None
    if segments.no_speech:
        print('[watch] whisper.cpp detected no speech.', file=sys.stderr)
    return segments
