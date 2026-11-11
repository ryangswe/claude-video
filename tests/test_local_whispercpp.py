"""whisper.cpp adapter: word regrouping, a fake whisper-cli child, and the installer."""
import json
import os
import sys
from pathlib import Path

import pytest
import config
import local_whispercpp as cpp
import setup
import whisper


def words(*items):
    return [(start, end, text) for start, end, text in items]


def test_regroup_joins_words_and_closes_sentences():
    out = cpp.regroup(words((0.5, 0.6, ''), (0.56, 0.84, ' Mr.'), (0.84, 1.22, ' Quilter'), (1.22, 1.33, ' is'),
                            (1.33, 5.29, ' the apostle of the middle classes and glad.'), (6.44, 6.54, ' Nor'),
                            (6.45, 6.73, ' is'), (6.73, 10.1, ' he.')))
    assert [s['text'] for s in out] == ['Mr. Quilter is the apostle of the middle classes and glad.', 'Nor is he.']
    assert out[0]['start'] == 0.56 and out[1]['start'] == 6.44


def spoken(word):
    return 0.2 + cpp.CHAR_SECONDS * len(word)


def test_stretched_punctuated_word_with_no_jump_closes_the_sentence_before_a_silence():
    # Real output from a short clip. "Mr." really belongs after the silence, but with no jump
    # before it the rule cannot tell an abbreviation from a sentence end: an accepted limit.
    out = cpp.regroup(words((1047.4, 1047.5, ' In'), (1048.3, 1048.75, ' guard,'), (1048.75, 1055.09, ' Mr.'),
                            (1055.09, 1110.6, ' Quilter'), (1110.6, 1110.73, ' is')))
    assert [s['text'] for s in out] == ['In guard, Mr.', 'Quilter is']
    assert out[0]['end'] == pytest.approx(1048.75 + spoken('Mr.'))
    assert out[1]['start'] == pytest.approx(1110.6 - spoken('Quilter'))


def test_long_bare_word_opens_the_speech_after_a_silence():
    out = cpp.regroup(words((9.38, 10.1, ' matter.'), (65.17, 71.27, ' He'), (71.27, 71.43, ' tells')))
    assert [s['text'] for s in out] == ['matter.', 'He tells']
    assert out[1]['start'] == pytest.approx(71.27 - spoken('He'))


def test_pauses_and_length_split_segments_and_starts_stay_ordered():
    out = cpp.regroup(words((0.0, 0.3, ' one'), (2.0, 2.3, ' two'), *[(3 + i * 0.1, 3.05 + i * 0.1, ' word')
                                                                       for i in range(60)]))
    assert out[0]['text'] == 'one' and out[1]['text'].startswith('two')
    assert all(len(s['text']) <= cpp.MAX_SEGMENT for s in out)
    assert [s['start'] for s in out] == sorted(s['start'] for s in out)


@pytest.fixture
def fake_cpp(tmp_path, monkeypatch):
    bin_dir = tmp_path / 'whisper cpp bin'
    bin_dir.mkdir()
    script = bin_dir / 'fake.py'
    script.write_text('''import json, os, pathlib, sys, time
mode = os.environ.get('WATCH_TEST_MODE', 'good')
if mode == 'slow': time.sleep(10)
args = sys.argv
assert '--vad' in args and args[args.index('-ml') + 1] == '1' and '-sow' in args
assert args[args.index('-mc') + 1] == '0'  # no carried text, so no repetition loops
assert pathlib.Path(args[args.index('-vm') + 1]).name == 'ggml-silero-v5.1.2.bin'
print('[00:00:00.000 --> 00:00:01.000]   a whole transcript line printed to stdout')
if mode == 'crash':
    print('ggml_metal: out of memory'); sys.exit(1)
if mode == 'crash_with_speech':
    print('[00:00:02.000 --> 00:00:03.000]   the download failed because the connection was offline'); sys.exit(1)
if mode == 'long_speech_crash':  # a transcript line longer than the kept tail
    print('[00:00:02.000 --> 00:00:03.000]  ' + ' the download stalled on a slow connection' * 500); sys.exit(1)
if mode == 'usage':  # like the real binary: the error, then about 5 KB of usage text, exit 0
    print("error: unknown language 'xx'")
    print('usage: whisper-cli [options] file0 file1 ...\\n' + ''.join(f'  -o{i:<4} N  [default] an option\\n'
                                                                    for i in range(160)))
    sys.exit(0)
base = args[args.index('-of') + 1]
if mode == 'malformed': pathlib.Path(base + '.json').write_text('{'); sys.exit(0)
lang = args[args.index('-l') + 1]
words = [] if mode == 'empty' else [
    {'offsets': {'from': 1000, 'to': 1400}, 'text': ' Hola'},
    {'offsets': {'from': 1400, 'to': 1900}, 'text': ' ' + lang + '.'},
]
data = json.dumps({'result': {'language': 'es'}, 'transcription': words}, ensure_ascii=False).encode()
if mode == 'badbytes':
    data = data.replace(b'Hola', b'Hol\\xc3')  # a word cut inside a UTF-8 character
pathlib.Path(base + '.json').write_bytes(data)
''', encoding='utf-8')
    exe = bin_dir / ('whisper-cli.cmd' if os.name == 'nt' else 'whisper-cli')
    if os.name == 'nt':
        exe.write_text(f'@"{sys.executable}" "%~dp0fake.py" %*\r\n')
    else:
        exe.write_text(f'#!{sys.executable}\n' + script.read_text())
        exe.chmod(0o755)
    cpp.model_dir().mkdir(parents=True)
    for path in (cpp.model_path('large-v3-turbo'), cpp.vad_path(), cpp.sentinel()):
        path.write_bytes(b'model')
    config.write_settings({'WATCH_WHISPERCPP_BIN': str(exe), 'WATCH_WHISPER_BACKEND': 'whispercpp'})
    audio = tmp_path / 'audio.with dots.mp3'
    audio.write_bytes(b'audio')
    return exe, audio


def test_child_output_is_regrouped_and_quiet(fake_cpp, capsys):
    segments = cpp.transcribe_audio(fake_cpp[1])
    assert segments == [{'start': 1.0, 'end': 1.9, 'text': 'Hola auto.'}] and segments.language == 'es'
    assert 'transcript line' not in capsys.readouterr().err
    assert not list(fake_cpp[1].parent.glob('watch-whispercpp-*'))


def test_language_hint_is_passed(fake_cpp, monkeypatch):
    monkeypatch.setenv('WATCH_WHISPERCPP_LANGUAGE', 'es')
    assert cpp.transcribe_audio(fake_cpp[1])[0]['text'] == 'Hola es.'


def test_empty_is_no_speech_and_bad_bytes_are_tolerated(fake_cpp, monkeypatch):
    monkeypatch.setenv('WATCH_TEST_MODE', 'empty')
    segments = cpp.transcribe_audio(fake_cpp[1])
    assert not segments and segments.no_speech
    monkeypatch.setenv('WATCH_TEST_MODE', 'badbytes')
    assert cpp.transcribe_audio(fake_cpp[1])


@pytest.mark.parametrize('mode,message', [('malformed', 'malformed'), ('crash', 'WATCH_WHISPERCPP_MODEL=small')])
def test_failures_name_whispercpp_and_keep_the_output_tail(fake_cpp, monkeypatch, mode, message):
    monkeypatch.setenv('WATCH_TEST_MODE', mode)
    with pytest.raises(SystemExit, match=message) as error:
        cpp.transcribe_audio(fake_cpp[1])
    assert 'whisper.cpp' in str(error.value)


def test_timeout_names_the_setting(fake_cpp, monkeypatch):
    monkeypatch.setenv('WATCH_TEST_MODE', 'slow')
    monkeypatch.setenv('WATCH_WHISPERCPP_TIMEOUT', '0.2')
    with pytest.raises(SystemExit, match='WATCH_WHISPERCPP_TIMEOUT'):
        cpp.transcribe_audio(fake_cpp[1])


def test_missing_models_and_binary(fake_cpp):
    exe, audio = fake_cpp
    cpp.vad_path().unlink()
    with pytest.raises(SystemExit, match='incomplete'):
        cpp.transcribe_audio(audio)
    exe.unlink()
    with pytest.raises(SystemExit, match='missing'):
        cpp.transcribe_audio(audio)


def test_route_never_resolves_a_cloud_key(fake_cpp, monkeypatch):
    exe, audio = fake_cpp
    monkeypatch.setattr(whisper, 'load_api_key', lambda *a, **k: pytest.fail('no cloud credentials for local inference'))
    monkeypatch.setattr(whisper, 'extract_audio', lambda *a: audio)
    assert whisper.transcribe_video('video.mp4', audio, backend='whispercpp')[1] == 'whispercpp'


@pytest.mark.parametrize('key,value', [('WATCH_WHISPERCPP_TIMEOUT', 'nan'), ('WATCH_WHISPERCPP_LANGUAGE', 'english'),
                                     ('WATCH_WHISPERCPP_LANGUAGE', 'e'),
                                     ('WATCH_WHISPERCPP_MODEL', 'medium')])
def test_invalid_settings(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(config.ConfigError):
        cpp.settings()


def test_status_reports_readiness(fake_cpp):
    status = setup._status()
    assert status['configured_backend'] == 'whispercpp' and status['whispercpp_ready'] and status['backend_ready']


def _fake_install(monkeypatch, commands, *, have_cli=True, brew_installs=False, has_vad=True):
    state = {'cli': have_cli}
    monkeypatch.setattr(setup, '_which', lambda name: None if name == 'whisper-cli' and not state['cli'] else '/fake/' + name)
    monkeypatch.setattr(setup.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(setup, '_whisper_cli_has_vad', lambda exe: has_vad)
    def step(cmd, label, advice=''):
        commands.append(cmd)
        if cmd[:2] == ['brew', 'install'] and brew_installs:
            state['cli'] = True
        if cmd[0] == 'curl':
            name = Path(cmd[cmd.index('-o') + 1]).name.removesuffix('.part')
            info = next(m for m in (*cpp.MODELS.values(), cpp.VAD_MODEL) if m[0] == name)
            Path(cmd[cmd.index('-o') + 1]).write_bytes(b'x')
            monkeypatch.setitem(sizes, name, info)
        if cmd[0] == 'ffmpeg':
            Path(cmd[-1]).write_bytes(b'audio')
    sizes = {}
    monkeypatch.setattr(setup, '_install_step', step)
    monkeypatch.setattr(setup, '_sha256', lambda path: next(m[2] for m in (*cpp.MODELS.values(), cpp.VAD_MODEL)
                                                            if m[0] == path.name.removesuffix('.part')))
    real_stat = Path.stat
    def stat(self, *a, **kw):
        result = real_stat(self, *a, **kw)
        match = [m for m in (*cpp.MODELS.values(), cpp.VAD_MODEL) if m[0] == self.name.removesuffix('.part')]
        return os.stat_result((result.st_mode, *result[1:6], match[0][3], *result[7:])) if match else result
    monkeypatch.setattr(Path, 'stat', stat)


def test_installer_downloads_verifies_and_selects_backend(monkeypatch):
    commands = []
    _fake_install(monkeypatch, commands)
    monkeypatch.setattr(cpp, 'transcribe_audio', lambda *a, **kw: [])
    assert setup.install_whispercpp() == 0
    assert sum(cmd[0] == 'curl' for cmd in commands) == 2
    assert cpp.sentinel().is_file() and cpp.model_path('large-v3-turbo').is_file() and cpp.vad_path().is_file()
    cfg = config.get_config()
    assert cfg['whisper_backend'] == 'whispercpp' and cfg['whispercpp_bin']
    assert not setup.is_first_run()
    commands.clear()
    setup.install_whispercpp()
    assert not any(cmd[0] == 'curl' for cmd in commands)


def test_installer_uses_homebrew_when_whisper_cli_is_missing(monkeypatch):
    commands = []
    _fake_install(monkeypatch, commands, have_cli=False)
    monkeypatch.setattr(cpp, 'transcribe_audio', lambda *a, **kw: [])
    with pytest.raises(SystemExit, match='brew install whisper.cpp'):
        setup.install_whispercpp()
    assert ['brew', 'install', 'whisper.cpp'] in commands
    assert not any(cmd[0] == 'curl' for cmd in commands)


def test_bad_checksum_discards_the_download(monkeypatch):
    _fake_install(monkeypatch, [])
    monkeypatch.setattr(setup, '_sha256', lambda path: '0' * 64)
    with pytest.raises(SystemExit, match='checksum'):
        setup.install_whispercpp()
    assert not list(cpp.model_dir().glob('*.part')) and not cpp.sentinel().exists()


def test_failed_warmup_never_marks_ready(monkeypatch):
    _fake_install(monkeypatch, [])
    def fail(*a, **kw):
        raise SystemExit('warmup failed')
    monkeypatch.setattr(cpp, 'transcribe_audio', fail)
    with pytest.raises(SystemExit, match='warmup failed'):
        setup.install_whispercpp()
    assert not cpp.sentinel().exists() and setup.is_first_run()
    assert config.get_config()['whisper_backend'] == 'auto'


def test_json_parse_keeps_language(tmp_path):
    path = tmp_path / 'out.json'
    path.write_text(json.dumps({'result': {'language': 'fr'},
                                'transcription': [{'offsets': {'from': 0, 'to': 500}, 'text': ' Bonjour.'}]}))
    segments = cpp.parse_output(path)
    assert segments == [{'start': 0.0, 'end': 0.5, 'text': 'Bonjour.'}] and segments.language == 'fr'


def test_long_punctuated_word_after_a_jump_opens_the_next_speech():
    # Real whisper.cpp output around a 60 s silence: the jump comes before "Mr.".
    out = cpp.regroup(words((1047.84, 1047.94, ' In'), (1048.31, 1048.82, ' guard,'), (1104.11, 1110.24, ' Mr.'),
                            (1110.24, 1110.61, ' Quilter'), (1110.61, 1110.73, ' is')))
    assert [s['text'] for s in out] == ['In guard,', 'Mr. Quilter is']
    assert out[0]['end'] == 1048.82 and out[1]['start'] == pytest.approx(1110.24 - spoken('Mr.'))


def test_stretched_punctuated_word_right_after_a_flushed_sentence_stays_before_the_silence():
    out = cpp.regroup(words((0, 5.29, ' the apostle of the middle classes and glad.'), (5.29, 65.0, ' Yes.'),
                            (65.0, 65.3, ' He'), (65.3, 65.6, ' tells.')))
    assert [s['text'] for s in out] == ['the apostle of the middle classes and glad. Yes.', 'He tells.']
    assert out[0]['end'] == pytest.approx(5.29 + spoken('Yes.')) and out[1]['start'] == 65.0


def test_no_space_language_phrases_keep_their_start(tmp_path):
    # Real whisper-cli output for a Chinese clip with speech at 3.0-12.7 s and 20.7-26.5 s.
    path = tmp_path / 'zh.json'
    path.write_text(json.dumps({'result': {'language': 'zh'}, 'transcription': [
        {'offsets': {'from': 3040, 'to': 12620}, 'text': ' 今天我们讨论视频分析的效率问题以及如何改进处理流程'},
        {'offsets': {'from': 19870, 'to': 25970}, 'text': ' 第二部分介绍语音识别和场景检测。'},
    ]}, ensure_ascii=False), encoding='utf-8')
    segments = cpp.parse_output(path)
    assert [s['start'] for s in segments] == [3.04, 19.87] and segments.language == 'zh'


def test_script_sets_the_rate_whatever_the_detected_language(tmp_path):
    path = tmp_path / 'ja.json'
    path.write_text(json.dumps({'result': {'language': 'en'}, 'transcription': [
        {'offsets': {'from': 1000, 'to': 7000}, 'text': ' これは十五文字くらいの日本語の文章です'}]}), encoding='utf-8')
    assert cpp.parse_output(path)[0]['start'] == 1.0


def test_stretched_phrase_after_a_silence_keeps_its_own_segment(tmp_path):
    # Real whisper-cli output: English, then Chinese A (speech 8.6-18.2 s), an 8 s silence and
    # Chinese B (speech 26.3-32.1 s). B's entry is stretched back over the silence.
    entries = [(0, 230, 'This'), (230, 340, ' is'), (340, 440, ' a'), (390, 620, ' long'), (620, 910, ' first'),
               (910, 1400, ' sentence'), (1400, 1640, ' that'), (1640, 1900, ' is'), (1900, 2180, ' clearly'),
               (2180, 2420, ' more'), (2420, 2740, ' than'), (2740, 3080, ' 40'), (3080, 3620, ' characters'),
               (3620, 3900, ' long.'), (4170, 4520, 'OK,'), (4520, 5500, ' thanks。'),
               (7130, 18210, '今天我们讨论机器学习的基本原理,包括监督学习和无监督学习的区别,然后我们会看几个具体的例子。'),
               (18310, 31660, '第二部分将则是神经网络的训练方法,以及如何选择合适的学习率。')]
    path = tmp_path / 'mix.json'
    path.write_text(json.dumps({'result': {'language': 'zh'}, 'transcription': [
        {'offsets': {'from': a, 'to': b}, 'text': t} for a, b, t in entries]}, ensure_ascii=False), encoding='utf-8')
    segments = cpp.parse_output(path)
    assert [s['text'][:4] for s in segments] == ['This', 'OK, ', '今天我们', '第二部分']
    assert segments[2]['start'] == 7.13 and segments[2]['end'] == 18.21
    assert 25.0 < segments[3]['start'] < 26.5 and segments[3]['end'] == 31.66


def test_exit_zero_without_json_shows_the_child_output(fake_cpp, monkeypatch):
    monkeypatch.setenv('WATCH_TEST_MODE', 'usage')
    with pytest.raises(SystemExit, match="unknown language 'xx'"):
        cpp.transcribe_audio(fake_cpp[1])


def test_a_cut_transcript_line_never_picks_the_failure_hint(fake_cpp, monkeypatch):
    monkeypatch.setenv('WATCH_TEST_MODE', 'long_speech_crash')
    with pytest.raises(SystemExit, match='exit 1') as error:
        cpp.transcribe_audio(fake_cpp[1])
    assert 'Model download' not in str(error.value)


def test_transcript_words_never_pick_the_failure_hint(fake_cpp, monkeypatch):
    monkeypatch.setenv('WATCH_TEST_MODE', 'crash_with_speech')
    with pytest.raises(SystemExit) as error:
        cpp.transcribe_audio(fake_cpp[1])
    assert 'Model download' not in str(error.value) and 'Memory' not in str(error.value)


def test_installer_finds_whisper_cli_after_homebrew_installs_it(monkeypatch):
    commands = []
    _fake_install(monkeypatch, commands, have_cli=False, brew_installs=True)
    monkeypatch.setattr(cpp, 'transcribe_audio', lambda *a, **kw: [])
    assert setup.install_whispercpp() == 0
    assert ['brew', 'install', 'whisper.cpp'] in commands
    assert config.get_config()['whispercpp_bin'] == os.path.abspath('/fake/whisper-cli')


def test_installer_rejects_a_whisper_cli_without_vad(monkeypatch):
    commands = []
    _fake_install(monkeypatch, commands, has_vad=False)
    with pytest.raises(SystemExit, match='brew upgrade whisper.cpp'):
        setup.install_whispercpp()
    assert not any(cmd[0] == 'curl' for cmd in commands)


def test_installer_prefers_a_configured_build(monkeypatch, tmp_path):
    custom = tmp_path / 'custom-whisper-cli'
    custom.write_text('')
    config.write_settings({'WATCH_WHISPERCPP_BIN': str(custom)})
    commands = []
    _fake_install(monkeypatch, commands, have_cli=False)
    monkeypatch.setattr(cpp, 'transcribe_audio', lambda *a, **kw: [])
    assert setup.install_whispercpp() == 0
    assert not any(cmd[0] == 'brew' for cmd in commands)
    assert config.get_config()['whispercpp_bin'] == str(custom)


def test_installer_falls_back_to_path_when_the_configured_build_lacks_vad(monkeypatch, tmp_path):
    old = tmp_path / 'old-whisper-cli'
    old.write_text('')
    config.write_settings({'WATCH_WHISPERCPP_BIN': str(old)})
    _fake_install(monkeypatch, [])
    monkeypatch.setattr(setup, '_whisper_cli_has_vad', lambda exe: exe != str(old))
    monkeypatch.setattr(cpp, 'transcribe_audio', lambda *a, **kw: [])
    assert setup.install_whispercpp() == 0
    assert config.get_config()['whispercpp_bin'] == os.path.abspath('/fake/whisper-cli')


def test_failed_curl_removes_the_partial_download(monkeypatch):
    _fake_install(monkeypatch, [])
    def step(cmd, label, advice=''):
        if cmd[0] == 'curl':
            Path(cmd[cmd.index('-o') + 1]).write_bytes(b'partial')
            raise SystemExit('Download failed (exit 56)')
    monkeypatch.setattr(setup, '_install_step', step)
    with pytest.raises(SystemExit, match='exit 56'):
        setup.install_whispercpp()
    assert not list(cpp.model_dir().glob('*.part'))
