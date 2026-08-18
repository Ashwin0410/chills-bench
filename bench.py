import contextlib
import inspect
import io
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
import traceback
import types
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from pydub import AudioSegment

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
DATA_DIR = Path(os.getenv("BENCH_DATA_DIR", "./data"))
AUDIO_DIR = DATA_DIR / "audio"
MUSIC_DIR = DATA_DIR / "music"
CONTEXT_DIR = DATA_DIR / "context"
DB_PATH = DATA_DIR / "bench.db"

for folder in (DATA_DIR, AUDIO_DIR, MUSIC_DIR, CONTEXT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Stimgen v2")

# how much context text can go into one generation, across all selected files
CONTEXT_CHAR_CAP = 80000

# the tags you can put on an experiment, the frontend colours them
TAGS = [
    "day 1 pick",
    "day 2 pick",
    "day 3 pick",
    "day 4 pick",
    "day 5 pick",
    "best so far",
    "chills",
    "close, needs work",
    "rejected",
]

SEED_VOICES = [
    ("Christian", "lMILJ9d29MrRXy9BIgcz"),
    ("Alicia", "OOk3INdXVLRmSaQoAX9D"),
]


# database

def db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def add_column(connection, table, column, definition):
    try:
        connection.execute(f"alter table {table} add column {column} {definition}")
    except Exception:
        pass


def init_db():
    connection = db()
    connection.execute("""
        create table if not exists experiments (
            id integer primary key autoincrement,
            created_at text not null,
            topic text default '',
            prompt_py text default '',
            mix_py text default '',
            model text default '',
            voice_id text default '',
            stability real default 0.5,
            style real default 0,
            boost integer default 1,
            music_filename text default '',
            music_file text default '',
            speech_text text default '',
            voice_file text default '',
            mix_file text default '',
            mix_source text default '',
            tts_provider text default 'elevenlabs',
            verdict text default '',
            comment text default '',
            parent_id integer
        )
    """)
    connection.execute("""
        create table if not exists saved_files (
            id integer primary key autoincrement,
            created_at text not null,
            kind text not null,
            name text not null,
            content text not null
        )
    """)
    connection.execute("""
        create table if not exists context_files (
            id integer primary key autoincrement,
            created_at text not null,
            name text not null,
            kind text not null,
            stored_name text not null,
            extracted text default '',
            chars integer default 0,
            note text default ''
        )
    """)
    connection.execute("""
        create table if not exists voices (
            id integer primary key autoincrement,
            created_at text not null,
            name text not null,
            voice_id text not null
        )
    """)

    # columns added after the first version, safe to run every start
    add_column(connection, "experiments", "mix_source", "text default ''")
    add_column(connection, "experiments", "tts_provider", "text default 'elevenlabs'")
    add_column(connection, "experiments", "title", "text default ''")
    add_column(connection, "experiments", "tag", "text default ''")
    add_column(connection, "experiments", "reflection", "text default ''")
    add_column(connection, "experiments", "protocol_id", "integer")
    add_column(connection, "experiments", "day_number", "integer default 1")
    add_column(connection, "experiments", "prior_id", "integer")
    add_column(connection, "experiments", "context_ids", "text default ''")
    add_column(connection, "experiments", "run_log", "text default ''")
    add_column(connection, "experiments", "validation", "text default ''")
    add_column(connection, "experiments", "word_count", "integer default 0")
    add_column(connection, "experiments", "music_gain_db", "real")
    add_column(connection, "experiments", "voice_lufs", "real")
    add_column(connection, "experiments", "sync_mode", "text default ''")
    add_column(connection, "experiments", "mix_profile", "text default ''")
    add_column(connection, "experiments", "prompt_source", "text default ''")

    # every old row belongs to a one day protocol of its own
    connection.execute("update experiments set protocol_id = id where protocol_id is null")
    connection.execute("update experiments set day_number = 1 where day_number is null")

    seeded = connection.execute("select count(*) as n from voices").fetchone()["n"]
    if seeded == 0:
        for name, voice_id in SEED_VOICES:
            connection.execute(
                "insert into voices (created_at, name, voice_id) values (?, ?, ?)",
                (now(), name, voice_id),
            )
        print(f"seeded {len(SEED_VOICES)} voices")

    connection.commit()
    connection.close()


def now():
    return datetime.now(timezone.utc).isoformat()


# audio helpers, ported from the main repo so pasted mix.py files work standalone

def load_audio(path):
    return AudioSegment.from_file(path)


def normalize_dbfs(segment, target_dbfs):
    return segment.apply_gain(target_dbfs - segment.dBFS)


def make_stereo(segment):
    return segment.set_channels(2)


def duration_ms(segment):
    return len(segment)


def measure_lufs(segment):
    # returns integrated loudness, or None when the file is too quiet to measure
    try:
        import pyloudnorm
        raw = np.frombuffer(segment.raw_data, dtype=np.int16).astype(np.float64) / 32768.0
        data = raw.reshape(-1, segment.channels)
        meter = pyloudnorm.Meter(segment.frame_rate)
        value = meter.integrated_loudness(data)
        if not np.isfinite(value) or value < -70.0:
            return None
        return float(value)
    except Exception:
        return None


def normalize_lufs(segment, target_lufs):
    measured = measure_lufs(segment)
    if measured is None:
        return segment
    return segment.apply_gain(target_lufs - measured)


def content_duration_sec(music_path):
    # same measurement as the repo's audio.py, scans back from the end for where sound stops
    try:
        segment = load_audio(music_path)
        total_ms = len(segment)
        if total_ms <= 0:
            return None
        window_ms = 500
        position = total_ms
        while position > 0:
            chunk = segment[max(0, position - window_ms):position]
            if chunk.rms > 0:
                full_scale = float(1 << (8 * chunk.sample_width - 1))
                rms_dbfs = 20.0 * math.log10(chunk.rms / full_scale)
            else:
                rms_dbfs = -120.0
            if rms_dbfs > -45.0:
                silent_tail = total_ms - position
                if silent_tail < 2000:
                    return total_ms / 1000.0
                print(f"music content ends at {position}ms, stripped {silent_tail}ms of tail")
                return position / 1000.0
            position -= window_ms
        return total_ms / 1000.0
    except Exception as error:
        print(f"content duration measurement failed: {error}")
        return None


def register_repo_modules():
    # pasted repo mix.py does "from ..utils.audio import ...", this makes that resolve
    app_module = types.ModuleType("app")
    app_module.__path__ = []
    utils_module = types.ModuleType("app.utils")
    utils_module.__path__ = []
    audio_module = types.ModuleType("app.utils.audio")
    audio_module.load_audio = load_audio
    audio_module.normalize_dbfs = normalize_dbfs
    audio_module.make_stereo = make_stereo
    audio_module.duration_ms = duration_ms
    app_module.utils = utils_module
    utils_module.audio = audio_module
    sys.modules.setdefault("app", app_module)
    sys.modules["app.utils"] = utils_module
    sys.modules["app.utils.audio"] = audio_module


register_repo_modules()
init_db()


# exec loaders for pasted files

def exec_pasted(code, module_name):
    namespace = {
        "__name__": module_name,
        "__package__": "app.services",
        "__builtins__": __builtins__,
    }
    exec(compile(code, module_name + ".py", "exec"), namespace)
    return namespace


PROMPT_NAMES = (
    "SYSTEM_PROMPT",
    "PROMPT",
    "MEDITATION_SYSTEM",
    "SPEECH1_SYSTEM",
    "JOURNAL_SYSTEM",
    "PLAN_SYSTEM",
)


def load_prompt_file(code, topic):
    # returns system prompt, user prompt, which variable it came from, and the
    # file's own validate function when it has one
    try:
        namespace = exec_pasted(code, "pasted_prompt")
    except Exception as error:
        raise HTTPException(400, f"prompt file failed to run: {error}")

    system_prompt = None
    source_name = ""
    for name in PROMPT_NAMES:
        value = namespace.get(name)
        if isinstance(value, str) and value.strip():
            system_prompt = value
            source_name = name
            break
    if system_prompt is None:
        # fall back to the largest string in the file
        candidates = [(k, v) for k, v in namespace.items()
                      if isinstance(v, str) and len(v) > 200 and not k.startswith("__")]
        if candidates:
            source_name, system_prompt = max(candidates, key=lambda pair: len(pair[1]))
            source_name = source_name + " (largest string, no known prompt name found)"
    if system_prompt is None:
        raise HTTPException(400, "no prompt string found in the pasted file, expected one of "
                                 + ", ".join(PROMPT_NAMES))

    builder = namespace.get("build_user_prompt")
    if callable(builder):
        try:
            user_prompt = call_builder(builder, topic)
            source_name += " plus build_user_prompt()"
        except Exception as error:
            raise HTTPException(400, f"build_user_prompt failed: {error}")
    else:
        user_prompt = topic if topic else "Write the piece now."

    validate = namespace.get("validate")
    if not callable(validate):
        validate = None

    return system_prompt, user_prompt, source_name, validate


def call_builder(builder, topic):
    # fill the first parameter with the topic, give sane values to the rest
    signature = inspect.signature(builder)
    arguments = {}
    first = True
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if first:
            arguments[name] = topic
            first = False
            continue
        if parameter.default is not parameter.empty:
            continue
        lowered = name.lower()
        if "word" in lowered:
            arguments[name] = 300
        elif "minute" in lowered:
            arguments[name] = 2
        elif lowered in ("challenges", "tips", "completed_actions"):
            arguments[name] = []
        else:
            arguments[name] = ""
    return builder(**arguments)


def run_validate(validate, script):
    # the prompt file's own quality check, never allowed to break generation
    if validate is None:
        return ""
    try:
        result = validate(script)
    except Exception as error:
        return f"validate() raised: {error}"
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    try:
        items = [str(item) for item in result]
    except Exception:
        return str(result)
    return "\n".join(items)


def load_mix_function(code):
    if not code.strip():
        return default_mix
    try:
        namespace = exec_pasted(code, "pasted_mix")
    except Exception as error:
        raise HTTPException(400, f"mix file failed to run: {error}")
    mix_function = namespace.get("mix")
    if not callable(mix_function):
        raise HTTPException(400, "no mix() function found in the pasted file")
    return mix_function


def default_mix(voice_path, music_path, out_path,
                music_premix_gain_db=-14.0, voice_target_lufs=None, **_):
    # plain fallback used when the mix box is empty, plays the full music track
    voice = make_stereo(load_audio(voice_path).set_frame_rate(44100))
    if voice_target_lufs is not None:
        voice = normalize_lufs(voice, float(voice_target_lufs))
    if music_path and Path(music_path).exists():
        music = make_stereo(load_audio(music_path).set_frame_rate(44100))
        music = music.apply_gain(float(music_premix_gain_db))
        if len(music) < len(voice):
            loops = len(voice) // len(music) + 1
            music = (music * loops)[:len(voice) + 3000]
        music = music.fade_out(2000)
        mixed = music.overlay(voice)
    else:
        mixed = voice
    mixed.export(out_path, format="mp3", bitrate="256k")
    return len(mixed)


DEFAULT_MIX_SOURCE = """# bench default mix, used because no mix file was pasted, plays the full music track
from pydub import AudioSegment
from pathlib import Path

def mix(voice_path, music_path, out_path,
        music_premix_gain_db=-14.0, voice_target_lufs=None, **_):
    voice = AudioSegment.from_file(voice_path).set_frame_rate(44100).set_channels(2)
    # voice_target_lufs normalization is applied by the bench when set
    if music_path and Path(music_path).exists():
        music = AudioSegment.from_file(music_path).set_frame_rate(44100).set_channels(2)
        music = music.apply_gain(float(music_premix_gain_db))
        if len(music) < len(voice):
            loops = len(voice) // len(music) + 1
            music = (music * loops)[:len(voice) + 3000]
        music = music.fade_out(2000)
        mixed = music.overlay(voice)
    else:
        mixed = voice
    mixed.export(out_path, format="mp3", bitrate="256k")
"""


def mix_source_for(mix_py, music_path):
    if mix_py.strip():
        return "pasted mix.py"
    if music_path:
        return "bench default"
    return "voice only"


def explicit_params(function):
    # names the function actually declares, ignoring **kwargs. mix_v45 ends in
    # **_ignored, so anything not declared is swallowed without an error and
    # settings would silently do nothing.
    try:
        signature = inspect.signature(function)
    except Exception:
        return set()
    return {
        name for name, parameter in signature.parameters.items()
        if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    }


def pre_gain_file(source_path, gain_db, scratch_dir, label):
    # used when the mix function has no gain argument of its own
    segment = load_audio(source_path)
    segment = segment.apply_gain(float(gain_db))
    target = Path(scratch_dir) / f"{label}_gained.wav"
    segment.export(target, format="wav")
    return str(target)


def run_mix(mix_function, voice_path, music_path, out_path, settings, log):
    # settings holds music_gain_db, voice_lufs, sync_mode, mix_profile, any may be None
    content_sec = None
    if music_path and Path(str(music_path)).exists():
        content_sec = content_duration_sec(music_path)
        log.append(f"music content duration {content_sec if content_sec is not None else 'not measured'}")

    accepted = explicit_params(mix_function)
    log.append("mix accepts: " + (", ".join(sorted(accepted)) if accepted else "nothing declared"))

    kwargs = {}
    dropped = []

    if content_sec is not None:
        if "content_duration_sec" in accepted:
            kwargs["content_duration_sec"] = content_sec
        else:
            dropped.append("content_duration_sec")

    music_gain = settings.get("music_gain_db")
    voice_lufs = settings.get("voice_lufs")
    sync_mode = settings.get("sync_mode")
    mix_profile = settings.get("mix_profile")

    if music_gain is not None:
        if "music_premix_gain_db" in accepted:
            kwargs["music_premix_gain_db"] = music_gain
            log.append(f"music gain {music_gain} dB passed as music_premix_gain_db")
        else:
            dropped.append("music_premix_gain_db")
    if voice_lufs is not None:
        if "voice_target_lufs" in accepted:
            kwargs["voice_target_lufs"] = voice_lufs
            log.append(f"voice level {voice_lufs} LUFS passed as voice_target_lufs")
        else:
            dropped.append("voice_target_lufs")
    if sync_mode:
        if "sync_mode" in accepted:
            kwargs["sync_mode"] = sync_mode
            log.append(f"sync mode {sync_mode}")
        else:
            dropped.append("sync_mode")
    if mix_profile:
        if "mix_profile" in accepted:
            kwargs["mix_profile"] = mix_profile
            log.append(f"mix profile {mix_profile}")
        else:
            dropped.append("mix_profile")

    scratch = tempfile.mkdtemp(prefix="bench_mix_")
    effective_voice = str(voice_path)
    effective_music = str(music_path) if music_path else ""

    try:
        # fallback path: the mix has no gain arguments, so gain the input files
        if music_gain is not None and "music_premix_gain_db" not in accepted and effective_music:
            effective_music = pre_gain_file(effective_music, music_gain, scratch, "music")
            log.append(f"music gain {music_gain} dB applied to the input file instead, "
                       "the mix may renormalize and cancel it")
        if voice_lufs is not None and "voice_target_lufs" not in accepted:
            segment = normalize_lufs(load_audio(effective_voice), float(voice_lufs))
            target = Path(scratch) / "voice_gained.wav"
            segment.export(target, format="wav")
            effective_voice = str(target)
            log.append(f"voice normalized to {voice_lufs} LUFS on the input file instead, "
                       "the mix may renormalize and cancel it")

        if dropped:
            log.append("this mix has no argument for: " + ", ".join(dropped))

        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                returned = mix_function(
                    voice_path=effective_voice,
                    music_path=effective_music,
                    out_path=str(out_path),
                    **kwargs,
                )
        except TypeError as error:
            # a mix that does not take keyword arguments at all
            log.append(f"keyword call rejected ({error}), retrying positionally")
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                returned = mix_function(effective_voice, effective_music, str(out_path))

        printed = captured.getvalue().strip()
        if printed:
            log.append("mix output:")
            for line in printed.splitlines():
                log.append("  " + line)
        if isinstance(returned, int):
            log.append(f"mix returned {returned} ms")

    except HTTPException:
        raise
    except Exception as error:
        printed = ""
        log.append("mix failed")
        log.append(traceback.format_exc().strip())
        shutil.rmtree(scratch, ignore_errors=True)
        raise HTTPException(500, f"mix failed: {error}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if not Path(out_path).exists():
        raise HTTPException(500, "mix ran but produced no output file")


def describe_audio(path, label, log):
    try:
        segment = load_audio(path)
        peak = segment.max_dBFS
        peak_text = f"{peak:.1f} dBFS" if peak != float("-inf") else "silent"
        lufs = measure_lufs(segment)
        lufs_text = f", {lufs:.1f} LUFS" if lufs is not None else ""
        log.append(f"{label}: {len(segment)} ms, peak {peak_text}{lufs_text}")
        return len(segment)
    except Exception as error:
        log.append(f"{label}: could not measure ({error})")
        return None


# claude

RETRYABLE = {429, 500, 502, 503, 504, 529}


def call_claude(model, system_prompt, user_prompt):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(503, "ANTHROPIC_API_KEY is not set on the server")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    last_error = None
    for attempt in range(1, 4):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = "".join(block.text for block in message.content if block.type == "text")
            return text.strip()
        except Exception as error:
            last_error = error
            status = getattr(error, "status_code", None)
            transient = status in RETRYABLE or (status is None and ("connect" in str(error).lower() or "timeout" in str(error).lower()))
            if transient and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            raise HTTPException(502, f"claude call failed: {error}")
    raise HTTPException(502, f"claude call failed: {last_error}")


# elevenlabs tts, same chunking and pause handling as production

SENTENCE_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")
PAUSE_TOKEN = "[pause]"
LONG_PAUSE_TOKEN = "[long pause]"
BREAK_TAG_RE = re.compile(r'<break\s+time="([0-9.]+)\s*(ms|s)"\s*/?\s*>', re.IGNORECASE)
BREAK_CLOSE_RE = re.compile(r"</\s*break\s*>", re.IGNORECASE)
MAX_CHARS = 3200
PAUSE_MS = 3000
LONG_PAUSE_MS = 6000
CHUNK_GAP_MS = 350


def split_chunks(text, max_chars=MAX_CHARS):
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    sentences = SENTENCE_SPLIT.split(text)
    chunks = []
    current = []
    current_length = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        add = len(sentence) + (1 if current_length else 0)
        if current_length + add <= max_chars:
            current.append(sentence)
            current_length += add
        else:
            if current:
                chunks.append(" ".join(current))
            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i:i + max_chars])
                current, current_length = [], 0
            else:
                current, current_length = [sentence], len(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def synth_chunk(text, voice_id, voice_settings, model_id="eleven_v3"):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "accept": "audio/mpeg", "Content-Type": "application/json"}
    payload = {"text": text, "model_id": model_id, "voice_settings": voice_settings}
    last_error = None
    for attempt in range(1, 4):
        try:
            response = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
            if response.status_code in (429, 500, 502, 503, 504):
                last_error = Exception(f"elevenlabs http {response.status_code}")
                if attempt < 3:
                    time.sleep(1.5 ** attempt)
                    continue
                response.raise_for_status()
            response.raise_for_status()
            buffer = io.BytesIO()
            for piece in response.iter_content(16384):
                if piece:
                    buffer.write(piece)
            buffer.seek(0)
            return AudioSegment.from_file(buffer, format="mp3")
        except requests.RequestException as error:
            last_error = error
            if attempt < 3:
                time.sleep(1.5 ** attempt)
                continue
            raise HTTPException(502, f"tts failed: {error}")
    raise HTTPException(502, f"tts failed: {last_error}")


def synth(text, voice_id, voice_settings, out_path, tts_provider="elevenlabs", log=None):
    if log is None:
        log = []
    if not ELEVENLABS_API_KEY:
        raise HTTPException(503, "ELEVENLABS_API_KEY is not set on the server")
    # every pause form becomes one ms sentinel: ssml breaks keep their duration,
    # [pause] is PAUSE_MS, [long pause] is LONG_PAUSE_MS
    raw = text.strip().replace("[breath]", " ")
    raw = BREAK_CLOSE_RE.sub(" ", raw)

    def break_to_sentinel(match):
        value = float(match.group(1))
        ms = int(value) if match.group(2).lower() == "ms" else int(value * 1000)
        return f" <<<BREAK:{ms}>>> "

    break_tags = len(BREAK_TAG_RE.findall(raw))
    long_pauses = raw.count(LONG_PAUSE_TOKEN)
    short_pauses = raw.count(PAUSE_TOKEN)

    raw = BREAK_TAG_RE.sub(break_to_sentinel, raw)
    raw = raw.replace(LONG_PAUSE_TOKEN, f" <<<BREAK:{LONG_PAUSE_MS}>>> ")
    raw = raw.replace(PAUSE_TOKEN, f" <<<BREAK:{PAUSE_MS}>>> ")

    log.append(f"tts engine {tts_provider}")
    log.append(f"markers found: {break_tags} break tags, {long_pauses} [long pause], {short_pauses} [pause]")

    native_breaks = 0
    if tts_provider == "eleven_v2":
        # v2 understands ssml natively, pauses up to 3s go back into the text
        # as real break tags so the model renders them, no split, no stitch.
        # longer pauses stay as inserted silence since v2 caps breaks at 3s.
        def native_or_keep(match):
            nonlocal native_breaks
            ms = int(match.group(1))
            if ms <= 3000:
                native_breaks += 1
                return f' <break time="{ms / 1000:.1f}s" /> '
            return match.group(0)

        raw = re.sub(r"<<<BREAK:(\d+)>>>", native_or_keep, raw)
        log.append(f"{native_breaks} pauses rendered natively by v2, no seam at those points")

    parts = re.split(r"(<<<BREAK:\d+>>>)", raw)

    segments = []
    spoke = False
    api_calls = 0
    inserted_silence_ms = 0
    for part in parts:
        if part.startswith("<<<BREAK:"):
            # every sentinel adds its own silence, so stacked pauses compound
            ms = int(part[len("<<<BREAK:"):-len(">>>")])
            segments.append(AudioSegment.silent(duration=ms, frame_rate=44100))
            inserted_silence_ms += ms
            continue
        part = part.strip()
        if not part:
            continue
        chunks = split_chunks(part)
        for chunk_index, chunk in enumerate(chunks):
            print(f"tts chunk {chunk_index + 1}/{len(chunks)}, {len(chunk)} chars")
            log.append(f"tts call {api_calls + 1}, {len(chunk)} chars")
            if chunk_index:
                segments.append(AudioSegment.silent(duration=CHUNK_GAP_MS, frame_rate=44100))
                inserted_silence_ms += CHUNK_GAP_MS
            if tts_provider == "eleven_v2":
                segments.append(synth_chunk(chunk, voice_id, voice_settings, "eleven_multilingual_v2"))
            else:
                segments.append(synth_chunk(chunk, voice_id, voice_settings))
            api_calls += 1
        spoke = True
    if not spoke:
        raise HTTPException(400, "speech text is empty after cleanup")

    seams = max(0, api_calls - 1)
    log.append(f"{api_calls} tts calls, {seams} seam points, {inserted_silence_ms} ms of inserted silence")
    if seams == 0:
        log.append("one call, no stitching")

    full = segments[0]
    for segment in segments[1:]:
        full += segment

    try:
        import noisereduce
        samples = np.array(full.get_array_of_samples(), dtype=np.float32)
        reduced = noisereduce.reduce_noise(y=samples, sr=full.frame_rate, stationary=True, prop_decrease=0.75)
        reduced_int = np.int16(np.clip(reduced, -32768, 32767))
        full = AudioSegment(data=reduced_int.tobytes(), sample_width=full.sample_width,
                            frame_rate=full.frame_rate, channels=full.channels)
        print("noise reduction applied")
        log.append("noise reduction applied")
    except Exception as error:
        print(f"noise reduction skipped: {error}")
        log.append(f"noise reduction skipped: {error}")

    full.export(out_path, format="wav")
    return out_path


# context files

def extract_text(path, name):
    lowered = name.lower()
    if lowered.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except Exception as error:
            raise HTTPException(500, f"pdf reader is not installed: {error}")
        try:
            reader = PdfReader(str(path))
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            return "\n\n".join(pages).strip(), f"{len(reader.pages)} pages"
        except Exception as error:
            raise HTTPException(400, f"could not read this pdf: {error}")
    if lowered.endswith(".docx"):
        try:
            import docx
        except Exception as error:
            raise HTTPException(500, f"docx reader is not installed: {error}")
        try:
            document = docx.Document(str(path))
            paragraphs = [p.text for p in document.paragraphs]
            return "\n".join(paragraphs).strip(), f"{len(paragraphs)} paragraphs"
        except Exception as error:
            raise HTTPException(400, f"could not read this docx: {error}")
    if lowered.endswith(".doc"):
        raise HTTPException(400, "old .doc files are not supported, open it and save as .docx")
    if lowered.endswith(".txt") or lowered.endswith(".md"):
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace").strip(), "plain text"
        except Exception as error:
            raise HTTPException(400, f"could not read this file: {error}")
    raise HTTPException(400, "supported types are pdf, docx, txt and md")


def build_context_block(connection, context_ids):
    # returns the text to prepend to the system prompt, plus log lines
    lines = []
    ids = [int(piece) for piece in str(context_ids).split(",") if piece.strip().isdigit()]
    if not ids:
        return "", lines, []

    rows = []
    for context_id in ids:
        row = connection.execute("select * from context_files where id = ?", (context_id,)).fetchone()
        if row:
            rows.append(row)

    if not rows:
        return "", lines, []

    reference = []
    examples = []
    used = 0
    truncated = []

    for row in rows:
        text = row["extracted"] or ""
        remaining = CONTEXT_CHAR_CAP - used
        if remaining <= 0:
            truncated.append(row["name"])
            lines.append(f"context {row['name']} skipped, character cap reached")
            continue
        if len(text) > remaining:
            text = text[:remaining]
            truncated.append(row["name"])
            lines.append(f"context {row['name']} truncated to {remaining} chars")
        used += len(text)
        entry = f"--- {row['name']} ---\n{text}"
        if row["kind"] == "example":
            examples.append(entry)
        else:
            reference.append(entry)
        lines.append(f"context {row['name']} used as {row['kind']}, {len(text)} chars")

    block = ""
    if reference:
        block += (
            "REFERENCE MATERIAL\n"
            "Background material to inform how you write. Do not quote it, do not mention it, "
            "and do not let its vocabulary override the instructions below.\n\n"
            + "\n\n".join(reference)
            + "\n\n"
        )
    if examples:
        block += (
            "EXAMPLE PIECES\n"
            "Examples of the style and quality to aim for. Match their approach and their level. "
            "Do not reuse their content, their images, or their sentences.\n\n"
            + "\n\n".join(examples)
            + "\n\n"
        )
    if block:
        lines.append(f"context total {used} chars of a {CONTEXT_CHAR_CAP} cap")
    return block, lines, [row["id"] for row in rows]


# protocol chaining

def chain_for(connection, prior_id):
    # walks back from prior_id to the start of the protocol, oldest first
    chain = []
    seen = set()
    current = prior_id
    while current:
        if current in seen:
            break
        seen.add(current)
        row = connection.execute("select * from experiments where id = ?", (current,)).fetchone()
        if not row:
            break
        chain.append(row)
        current = row["prior_id"]
    chain.reverse()
    return chain


def build_prior_block(chain):
    # the format the days 2 to 5 prompt asks for: every meditation already given,
    # in order, with their reflections
    if not chain:
        return "", []
    lines = []
    parts = []
    for index, row in enumerate(chain, start=1):
        speech = (row["speech_text"] or "").strip()
        if not speech:
            lines.append(f"day {index} (experiment {row['id']}) has no text, skipped")
            continue
        parts.append(f"MEDITATION {index}\n{speech}")
        reflection = (row["reflection"] or "").strip()
        if reflection:
            parts.append(f"REFLECTION {index}\n{reflection}")
        else:
            parts.append(f"REFLECTION {index}\nnone given")
        lines.append(f"day {index} (experiment {row['id']}) included, {len(speech.split())} words, "
                     f"reflection {'yes' if reflection else 'no'}")
    if not parts:
        return "", lines
    return "\n\n" + "\n\n".join(parts), lines


def short_model(model):
    if not model:
        return ""
    lowered = model.lower()
    for name in ("opus", "sonnet", "haiku"):
        if name in lowered:
            return name
    return model


def voice_name_for(connection, voice_id):
    if not voice_id:
        return ""
    row = connection.execute("select name from voices where voice_id = ?", (voice_id,)).fetchone()
    return row["name"].lower() if row else ""


def build_title(connection, topic, day_number, model, voice_id):
    first_line = (topic or "").strip().splitlines()[0] if (topic or "").strip() else ""
    first_line = first_line.strip()
    if len(first_line) > 40:
        first_line = first_line[:40].rstrip() + "..."
    pieces = []
    if first_line:
        pieces.append(first_line)
    pieces.append(f"day {day_number or 1}")
    model_name = short_model(model)
    if model_name:
        pieces.append(model_name)
    voice = voice_name_for(connection, voice_id)
    if voice:
        pieces.append(voice)
    return ", ".join(pieces)


# experiment helpers

def experiment_row(row):
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "title": row["title"] or "",
        "topic": row["topic"],
        "prompt_py": row["prompt_py"],
        "mix_py": row["mix_py"],
        "model": row["model"],
        "voice_id": row["voice_id"],
        "stability": row["stability"],
        "style": row["style"],
        "boost": bool(row["boost"]),
        "music_filename": row["music_filename"],
        "speech_text": row["speech_text"],
        "voice_url": f"/api/bench/audio/{row['voice_file']}" if row["voice_file"] else None,
        "mix_url": f"/api/bench/audio/{row['mix_file']}" if row["mix_file"] else None,
        "mix_source": row["mix_source"] or "",
        "tts_provider": row["tts_provider"] or "elevenlabs",
        "verdict": row["verdict"],
        "comment": row["comment"],
        "reflection": row["reflection"] or "",
        "tag": row["tag"] or "",
        "parent_id": row["parent_id"],
        "protocol_id": row["protocol_id"] or row["id"],
        "day_number": row["day_number"] or 1,
        "prior_id": row["prior_id"],
        "context_ids": row["context_ids"] or "",
        "run_log": row["run_log"] or "",
        "validation": row["validation"] or "",
        "word_count": row["word_count"] or 0,
        "music_gain_db": row["music_gain_db"],
        "voice_lufs": row["voice_lufs"],
        "sync_mode": row["sync_mode"] or "",
        "mix_profile": row["mix_profile"] or "",
        "prompt_source": row["prompt_source"] or "",
    }


def save_upload(upload, destination):
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as handle:
        while True:
            piece = upload.file.read(1024 * 1024)
            if not piece:
                break
            handle.write(piece)


def fetch_experiment(connection, experiment_id):
    return connection.execute("select * from experiments where id = ?", (experiment_id,)).fetchone()


def optional_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        raise HTTPException(400, f"expected a number, got {text}")


def file_still_used(connection, field, value, exclude_id):
    if not value:
        return False
    row = connection.execute(
        f"select count(*) as n from experiments where {field} = ? and id != ?",
        (value, exclude_id),
    ).fetchone()
    return row["n"] > 0


# endpoints

class WriteReq(BaseModel):
    topic: str = ""
    prompt_py: str = ""
    model: str = "claude-sonnet-4-6"
    context_ids: str = ""
    prior_id: int = 0


@app.post("/api/bench/write")
def write_answer(req: WriteReq):
    if not req.prompt_py.strip():
        raise HTTPException(400, "paste a prompt file first")

    log = [f"write started {now()}"]
    connection = db()
    try:
        system_prompt, user_prompt, source_name, validate = load_prompt_file(
            req.prompt_py, req.topic.strip()
        )
        log.append(f"prompt taken from {source_name}")
        log.append(f"model {req.model}")

        context_block, context_lines, used_ids = build_context_block(connection, req.context_ids)
        log.extend(context_lines)
        if context_block:
            system_prompt = context_block + system_prompt

        protocol_id = None
        day_number = 1
        prior_id = req.prior_id or None
        if prior_id:
            prior = fetch_experiment(connection, prior_id)
            if not prior:
                raise HTTPException(404, "the experiment you are continuing from was not found")
            chain = chain_for(connection, prior_id)
            prior_block, prior_lines = build_prior_block(chain)
            log.extend(prior_lines)
            user_prompt = user_prompt + prior_block
            protocol_id = prior["protocol_id"] or prior["id"]
            day_number = (prior["day_number"] or 1) + 1
            log.append(f"continuing protocol {protocol_id}, this is day {day_number}")

        log.append(f"system prompt {len(system_prompt)} chars, user prompt {len(user_prompt)} chars")

        speech = call_claude(req.model, system_prompt, user_prompt)
        if not speech:
            raise HTTPException(502, "claude returned empty text")

        word_count = len(speech.split())
        log.append(f"claude returned {word_count} words, {len(speech)} chars")

        validation = run_validate(validate, speech)
        if validate is None:
            log.append("prompt file has no validate(), no check run")
        elif validation:
            log.append("validate() found problems:")
            for line in validation.splitlines():
                log.append("  " + line)
        else:
            log.append("validate() found no problems")

        # save on write: every generation becomes a card, audio empty until make runs
        cursor = connection.execute(
            "insert into experiments (created_at, topic, prompt_py, model, speech_text, "
            "context_ids, prior_id, day_number, run_log, validation, word_count, prompt_source) "
            "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (now(), req.topic.strip(), req.prompt_py, req.model, speech,
             ",".join(str(i) for i in used_ids), prior_id, day_number,
             "\n".join(log), validation, word_count, source_name),
        )
        experiment_id = cursor.lastrowid
        if protocol_id is None:
            protocol_id = experiment_id
        title = build_title(connection, req.topic.strip(), day_number, req.model, "")
        connection.execute(
            "update experiments set protocol_id = ?, title = ? where id = ?",
            (protocol_id, title, experiment_id),
        )
        connection.commit()
        print(f"experiment {experiment_id} text saved")
        return {
            "speech": speech,
            "experiment_id": experiment_id,
            "word_count": word_count,
            "validation": validation,
            "prompt_source": source_name,
            "day_number": day_number,
            "run_log": "\n".join(log),
        }
    finally:
        connection.close()


def attach_or_create(connection, experiment_id, topic, prompt_py, model, speech):
    # attach audio to the write card if it exists and has no audio yet, else new card
    if experiment_id:
        row = fetch_experiment(connection, experiment_id)
        if row and not row["voice_file"] and not row["mix_file"]:
            return experiment_id, True
    cursor = connection.execute(
        "insert into experiments (created_at, topic, prompt_py, model, speech_text) values (?, ?, ?, ?, ?)",
        (now(), topic, prompt_py, model, speech),
    )
    new_id = cursor.lastrowid
    connection.execute(
        "update experiments set protocol_id = ? where id = ?", (new_id, new_id)
    )
    connection.commit()
    return new_id, False


@app.post("/api/bench/make")
def make_mp3(
    topic: str = Form(""),
    speech: str = Form(...),
    prompt_py: str = Form(""),
    mix_py: str = Form(""),
    model: str = Form(""),
    voice_id: str = Form(...),
    stability: float = Form(0.5),
    style: float = Form(0.0),
    boost: bool = Form(True),
    tts_provider: str = Form("elevenlabs"),
    experiment_id: int = Form(0),
    voice_only: bool = Form(False),
    music_gain_db: str = Form(""),
    voice_lufs: str = Form(""),
    sync_mode: str = Form(""),
    mix_profile: str = Form(""),
    music: UploadFile | None = File(default=None),
):
    tts_provider = tts_provider.strip() or "elevenlabs"
    if tts_provider not in ("elevenlabs", "eleven_v2"):
        raise HTTPException(400, "tts provider must be elevenlabs or eleven_v2")
    speech = speech.strip()
    voice_id = voice_id.strip()
    if not speech:
        raise HTTPException(400, "speech text is empty")
    if not voice_id:
        raise HTTPException(400, "voice id is empty")

    settings = {
        "music_gain_db": optional_float(music_gain_db),
        "voice_lufs": optional_float(voice_lufs),
        "sync_mode": sync_mode.strip(),
        "mix_profile": mix_profile.strip(),
    }

    if voice_only:
        mix_py = ""
        music = None

    mix_function = load_mix_function(mix_py)

    log = [f"make started {now()}"]
    connection = db()
    target_id, attached = attach_or_create(connection, experiment_id, topic.strip(), prompt_py, model, speech)
    if attached:
        existing = fetch_experiment(connection, target_id)
        if existing and existing["run_log"]:
            log = existing["run_log"].splitlines() + ["", f"make started {now()}"]

    try:
        music_path = ""
        music_filename = ""
        music_rel = ""
        if music is not None and music.filename:
            # keep the original filename inside a per experiment folder, a prefix
            # would break mix files that read the name to pick a profile
            music_filename = Path(music.filename).name
            music_rel = f"{target_id}/{music_filename}"
            music_path = MUSIC_DIR / music_rel
            save_upload(music, music_path)
            log.append(f"music uploaded as {music_rel}")

        source = mix_source_for(mix_py, music_path)
        log.append(f"mix source {source}")

        voice_settings = {"stability": stability, "similarity_boost": 0.7, "style": style, "use_speaker_boost": boost}
        log.append(f"voice {voice_id}, stability {stability}, style {style}, "
                   f"boost {'on' if boost else 'off'}")

        voice_file = f"{target_id}_voice.wav"
        synth(speech, voice_id, voice_settings, str(AUDIO_DIR / voice_file), tts_provider, log)
        print(f"experiment {target_id} voice saved")
        describe_audio(AUDIO_DIR / voice_file, "voice", log)

        if voice_only:
            mix_file = ""
            log.append("voice only, no mix run")
        else:
            if music_path:
                describe_audio(music_path, "music", log)
            mix_file = f"{target_id}_mix.mp3"
            run_mix(mix_function, AUDIO_DIR / voice_file, music_path, AUDIO_DIR / mix_file, settings, log)
            print(f"experiment {target_id} mix saved, {source}")
            describe_audio(AUDIO_DIR / mix_file, "output", log)

        row_now = fetch_experiment(connection, target_id)
        title = (row_now["title"] or "") if row_now else ""
        if not title:
            day_number = (row_now["day_number"] if row_now else 1) or 1
            title = build_title(connection, topic.strip(), day_number, model, voice_id)

        connection.execute(
            "update experiments set topic = ?, prompt_py = ?, mix_py = ?, model = ?, voice_id = ?, "
            "stability = ?, style = ?, boost = ?, speech_text = ?, music_filename = ?, music_file = ?, "
            "voice_file = ?, mix_file = ?, mix_source = ?, tts_provider = ?, run_log = ?, "
            "word_count = ?, music_gain_db = ?, voice_lufs = ?, sync_mode = ?, mix_profile = ?, "
            "title = ? where id = ?",
            (topic.strip(), prompt_py, mix_py, model, voice_id, stability, style, int(boost), speech,
             music_filename, music_rel, voice_file, mix_file, source, tts_provider,
             "\n".join(log), len(speech.split()), settings["music_gain_db"], settings["voice_lufs"],
             settings["sync_mode"], settings["mix_profile"], title, target_id),
        )
        connection.commit()
        row = fetch_experiment(connection, target_id)
        return experiment_row(row)
    except HTTPException:
        if not attached:
            connection.execute("delete from experiments where id = ?", (target_id,))
            connection.commit()
        else:
            connection.execute("update experiments set run_log = ? where id = ?",
                               ("\n".join(log), target_id))
            connection.commit()
        raise
    except Exception as error:
        log.append(traceback.format_exc().strip())
        if not attached:
            connection.execute("delete from experiments where id = ?", (target_id,))
            connection.commit()
        else:
            connection.execute("update experiments set run_log = ? where id = ?",
                               ("\n".join(log), target_id))
            connection.commit()
        raise HTTPException(500, f"make failed: {error}")
    finally:
        connection.close()


@app.post("/api/bench/remix/{experiment_id}")
def remix(
    experiment_id: int,
    mix_py: str = Form(""),
    music_gain_db: str = Form(""),
    voice_lufs: str = Form(""),
    sync_mode: str = Form(""),
    mix_profile: str = Form(""),
    music: UploadFile | None = File(default=None),
):
    settings = {
        "music_gain_db": optional_float(music_gain_db),
        "voice_lufs": optional_float(voice_lufs),
        "sync_mode": sync_mode.strip(),
        "mix_profile": mix_profile.strip(),
    }

    connection = db()
    parent = fetch_experiment(connection, experiment_id)
    if not parent:
        connection.close()
        raise HTTPException(404, "experiment not found")
    if not parent["voice_file"] or not (AUDIO_DIR / parent["voice_file"]).exists():
        connection.close()
        raise HTTPException(400, "no stored voice file for this experiment")

    mix_function = load_mix_function(mix_py)

    log = [f"remix of experiment {experiment_id} started {now()}"]

    music_path = ""
    music_filename = parent["music_filename"]
    music_rel = ""
    reuse_parent_music = False
    if music is not None and music.filename:
        music_filename = Path(music.filename).name
    elif parent["music_file"] and (MUSIC_DIR / parent["music_file"]).exists():
        music_path = MUSIC_DIR / parent["music_file"]
        music_rel = parent["music_file"]
        reuse_parent_music = True
        log.append(f"reusing the music from experiment {experiment_id}")

    cursor = connection.execute(
        "insert into experiments (created_at, topic, prompt_py, mix_py, model, voice_id, stability, style, boost, "
        "speech_text, voice_file, parent_id, protocol_id, day_number, prior_id, reflection, word_count, "
        "validation, prompt_source, context_ids) "
        "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (now(), parent["topic"], parent["prompt_py"], mix_py, parent["model"], parent["voice_id"],
         parent["stability"], parent["style"], parent["boost"], parent["speech_text"],
         parent["voice_file"], experiment_id, parent["protocol_id"] or parent["id"],
         parent["day_number"] or 1, parent["prior_id"], parent["reflection"] or "",
         parent["word_count"] or 0, parent["validation"] or "", parent["prompt_source"] or "",
         parent["context_ids"] or ""),
    )
    new_id = cursor.lastrowid
    connection.commit()

    try:
        if music is not None and music.filename:
            music_rel = f"{new_id}/{music_filename}"
            music_path = MUSIC_DIR / music_rel
            save_upload(music, music_path)
            log.append(f"new music uploaded as {music_rel}")

        source = mix_source_for(mix_py, music_path)
        log.append(f"mix source {source}")

        describe_audio(AUDIO_DIR / parent["voice_file"], "voice", log)
        if music_path:
            describe_audio(music_path, "music", log)

        mix_file = f"{new_id}_mix.mp3"
        run_mix(mix_function, AUDIO_DIR / parent["voice_file"], music_path,
                AUDIO_DIR / mix_file, settings, log)
        print(f"experiment {new_id} remix of {experiment_id} saved, {source}")
        describe_audio(AUDIO_DIR / mix_file, "output", log)

        title = build_title(connection, parent["topic"], parent["day_number"] or 1,
                            parent["model"], parent["voice_id"])
        title = f"{title}, remix"

        connection.execute(
            "update experiments set music_filename = ?, music_file = ?, mix_file = ?, mix_source = ?, "
            "run_log = ?, music_gain_db = ?, voice_lufs = ?, sync_mode = ?, mix_profile = ?, "
            "title = ? where id = ?",
            (music_filename, music_rel, mix_file, source, "\n".join(log),
             settings["music_gain_db"], settings["voice_lufs"], settings["sync_mode"],
             settings["mix_profile"], title, new_id),
        )
        connection.commit()
        row = fetch_experiment(connection, new_id)
        return experiment_row(row)
    except HTTPException:
        if not reuse_parent_music and music_rel:
            shutil.rmtree(MUSIC_DIR / str(new_id), ignore_errors=True)
        connection.execute("delete from experiments where id = ?", (new_id,))
        connection.commit()
        raise
    except Exception as error:
        if not reuse_parent_music and music_rel:
            shutil.rmtree(MUSIC_DIR / str(new_id), ignore_errors=True)
        connection.execute("delete from experiments where id = ?", (new_id,))
        connection.commit()
        raise HTTPException(500, f"remix failed: {error}")
    finally:
        connection.close()


@app.delete("/api/bench/experiments/{experiment_id}")
def delete_experiment(experiment_id: int):
    connection = db()
    row = fetch_experiment(connection, experiment_id)
    if not row:
        connection.close()
        raise HTTPException(404, "experiment not found")

    # remixes share the parent's voice file, and reused music is shared too, so
    # only remove a file when no other row still points at it
    removed = []
    kept = []
    for field, folder in (("voice_file", AUDIO_DIR), ("mix_file", AUDIO_DIR), ("music_file", MUSIC_DIR)):
        name = row[field]
        if not name:
            continue
        if file_still_used(connection, field, name, experiment_id):
            kept.append(name)
            continue
        path = folder / name
        if path.exists():
            path.unlink()
            removed.append(name)
        if field == "music_file" and "/" in name:
            parent_folder = (MUSIC_DIR / name).parent
            try:
                if parent_folder.exists() and not any(parent_folder.iterdir()):
                    parent_folder.rmdir()
            except Exception:
                pass

    # anything that pointed at this row loses the link but keeps its own audio
    connection.execute("update experiments set prior_id = null where prior_id = ?", (experiment_id,))
    connection.execute("update experiments set parent_id = null where parent_id = ?", (experiment_id,))
    connection.execute("delete from experiments where id = ?", (experiment_id,))
    connection.commit()
    connection.close()
    print(f"deleted experiment {experiment_id}, removed {len(removed)} files, kept {len(kept)} shared")
    return {"status": "ok", "removed": removed, "kept_shared": kept}


@app.get("/api/bench/experiments")
def list_experiments():
    connection = db()
    rows = connection.execute("select * from experiments order by id desc").fetchall()
    connection.close()
    return {"experiments": [experiment_row(row) for row in rows], "tags": TAGS}


class VerdictReq(BaseModel):
    verdict: str = ""
    comment: str = ""


@app.post("/api/bench/experiments/{experiment_id}/verdict")
def set_verdict(experiment_id: int, req: VerdictReq):
    if req.verdict not in ("worked", "did not work", ""):
        raise HTTPException(400, "verdict must be worked, did not work, or empty")
    connection = db()
    row = fetch_experiment(connection, experiment_id)
    if not row:
        connection.close()
        raise HTTPException(404, "experiment not found")
    connection.execute("update experiments set verdict = ?, comment = ? where id = ?",
                       (req.verdict, req.comment.strip(), experiment_id))
    connection.commit()
    connection.close()
    return {"status": "ok"}


class TagReq(BaseModel):
    tag: str = ""


@app.post("/api/bench/experiments/{experiment_id}/tag")
def set_tag(experiment_id: int, req: TagReq):
    tag = req.tag.strip()
    if tag and tag not in TAGS:
        raise HTTPException(400, "unknown tag")
    connection = db()
    row = fetch_experiment(connection, experiment_id)
    if not row:
        connection.close()
        raise HTTPException(404, "experiment not found")
    connection.execute("update experiments set tag = ? where id = ?", (tag, experiment_id))
    connection.commit()
    connection.close()
    return {"status": "ok", "tag": tag}


class ReflectionReq(BaseModel):
    reflection: str = ""


@app.post("/api/bench/experiments/{experiment_id}/reflection")
def set_reflection(experiment_id: int, req: ReflectionReq):
    connection = db()
    row = fetch_experiment(connection, experiment_id)
    if not row:
        connection.close()
        raise HTTPException(404, "experiment not found")
    connection.execute("update experiments set reflection = ? where id = ?",
                       (req.reflection.strip(), experiment_id))
    connection.commit()
    connection.close()
    return {"status": "ok"}


class TitleReq(BaseModel):
    title: str = ""


@app.post("/api/bench/experiments/{experiment_id}/title")
def set_title(experiment_id: int, req: TitleReq):
    connection = db()
    row = fetch_experiment(connection, experiment_id)
    if not row:
        connection.close()
        raise HTTPException(404, "experiment not found")
    connection.execute("update experiments set title = ? where id = ?",
                       (req.title.strip()[:200], experiment_id))
    connection.commit()
    connection.close()
    return {"status": "ok"}


@app.get("/api/bench/experiments/{experiment_id}/chain")
def get_chain(experiment_id: int):
    connection = db()
    row = fetch_experiment(connection, experiment_id)
    if not row:
        connection.close()
        raise HTTPException(404, "experiment not found")
    protocol_id = row["protocol_id"] or row["id"]
    rows = connection.execute(
        "select * from experiments where protocol_id = ? order by day_number, id",
        (protocol_id,),
    ).fetchall()
    connection.close()
    return {"protocol_id": protocol_id, "experiments": [experiment_row(item) for item in rows]}


# saved file library

class SaveFileReq(BaseModel):
    kind: str
    name: str
    content: str


@app.get("/api/bench/files")
def list_files(kind: str = ""):
    connection = db()
    if kind:
        rows = connection.execute("select id, created_at, kind, name from saved_files where kind = ? order by id desc", (kind,)).fetchall()
    else:
        rows = connection.execute("select id, created_at, kind, name from saved_files order by id desc").fetchall()
    connection.close()
    return {"files": [dict(row) for row in rows]}


@app.get("/api/bench/files/{file_id}")
def get_file(file_id: int):
    connection = db()
    row = connection.execute("select * from saved_files where id = ?", (file_id,)).fetchone()
    connection.close()
    if not row:
        raise HTTPException(404, "file not found")
    return dict(row)


@app.post("/api/bench/files")
def save_file(req: SaveFileReq):
    if req.kind not in ("prompt", "mix"):
        raise HTTPException(400, "kind must be prompt or mix")
    if not req.name.strip():
        raise HTTPException(400, "name required")
    if not req.content.strip():
        raise HTTPException(400, "content is empty")
    connection = db()
    cursor = connection.execute(
        "insert into saved_files (created_at, kind, name, content) values (?, ?, ?, ?)",
        (now(), req.kind, req.name.strip()[:100], req.content),
    )
    connection.commit()
    file_id = cursor.lastrowid
    connection.close()
    print(f"saved {req.kind} file {file_id}: {req.name.strip()[:100]}")
    return {"id": file_id}


@app.delete("/api/bench/files/{file_id}")
def delete_file(file_id: int):
    connection = db()
    connection.execute("delete from saved_files where id = ?", (file_id,))
    connection.commit()
    connection.close()
    return {"status": "ok"}


# context library

@app.get("/api/bench/context")
def list_context():
    connection = db()
    rows = connection.execute(
        "select id, created_at, name, kind, chars, note from context_files order by id desc"
    ).fetchall()
    connection.close()
    return {"context": [dict(row) for row in rows], "cap": CONTEXT_CHAR_CAP}


@app.get("/api/bench/context/{context_id}")
def get_context(context_id: int):
    connection = db()
    row = connection.execute("select * from context_files where id = ?", (context_id,)).fetchone()
    connection.close()
    if not row:
        raise HTTPException(404, "context file not found")
    data = dict(row)
    # only the start of the text, the whole book is not useful on screen
    data["preview"] = (row["extracted"] or "")[:4000]
    data.pop("extracted", None)
    return data


@app.post("/api/bench/context")
def upload_context(
    kind: str = Form("reference"),
    file: UploadFile = File(...),
):
    kind = kind.strip().lower()
    if kind not in ("reference", "example"):
        raise HTTPException(400, "kind must be reference or example")
    if not file or not file.filename:
        raise HTTPException(400, "no file uploaded")

    name = Path(file.filename).name
    connection = db()
    cursor = connection.execute(
        "insert into context_files (created_at, name, kind, stored_name) values (?, ?, ?, ?)",
        (now(), name, kind, ""),
    )
    context_id = cursor.lastrowid
    connection.commit()

    stored_name = f"{context_id}_{name}"
    stored_path = CONTEXT_DIR / stored_name
    try:
        save_upload(file, stored_path)
        text, note = extract_text(stored_path, name)
        if not text.strip():
            raise HTTPException(400, "no text could be read from this file, it may be a scan")
        connection.execute(
            "update context_files set stored_name = ?, extracted = ?, chars = ?, note = ? where id = ?",
            (stored_name, text, len(text), note, context_id),
        )
        connection.commit()
        print(f"context {context_id} saved: {name}, {len(text)} chars")
        row = connection.execute(
            "select id, created_at, name, kind, chars, note from context_files where id = ?",
            (context_id,),
        ).fetchone()
        return dict(row)
    except HTTPException:
        connection.execute("delete from context_files where id = ?", (context_id,))
        connection.commit()
        if stored_path.exists():
            stored_path.unlink()
        raise
    except Exception as error:
        connection.execute("delete from context_files where id = ?", (context_id,))
        connection.commit()
        if stored_path.exists():
            stored_path.unlink()
        raise HTTPException(500, f"context upload failed: {error}")
    finally:
        connection.close()


@app.delete("/api/bench/context/{context_id}")
def delete_context(context_id: int):
    connection = db()
    row = connection.execute("select * from context_files where id = ?", (context_id,)).fetchone()
    if not row:
        connection.close()
        raise HTTPException(404, "context file not found")
    if row["stored_name"]:
        path = CONTEXT_DIR / row["stored_name"]
        if path.exists():
            path.unlink()
    connection.execute("delete from context_files where id = ?", (context_id,))
    connection.commit()
    connection.close()
    return {"status": "ok"}


# voices

class VoiceReq(BaseModel):
    name: str
    voice_id: str


@app.get("/api/bench/voices")
def list_voices():
    connection = db()
    rows = connection.execute("select * from voices order by name").fetchall()
    connection.close()
    return {"voices": [dict(row) for row in rows]}


@app.post("/api/bench/voices")
def save_voice(req: VoiceReq):
    name = req.name.strip()[:60]
    voice_id = req.voice_id.strip()
    if not name:
        raise HTTPException(400, "name required")
    if not voice_id:
        raise HTTPException(400, "voice id required")
    connection = db()
    existing = connection.execute("select id from voices where voice_id = ?", (voice_id,)).fetchone()
    if existing:
        connection.execute("update voices set name = ? where id = ?", (name, existing["id"]))
        voice_row_id = existing["id"]
    else:
        cursor = connection.execute(
            "insert into voices (created_at, name, voice_id) values (?, ?, ?)",
            (now(), name, voice_id),
        )
        voice_row_id = cursor.lastrowid
    connection.commit()
    connection.close()
    return {"id": voice_row_id, "name": name, "voice_id": voice_id}


@app.delete("/api/bench/voices/{voice_row_id}")
def delete_voice(voice_row_id: int):
    connection = db()
    connection.execute("delete from voices where id = ?", (voice_row_id,))
    connection.commit()
    connection.close()
    return {"status": "ok"}


@app.get("/api/bench/experiments/{experiment_id}/zip")
def get_zip(experiment_id: int):
    connection = db()
    row = fetch_experiment(connection, experiment_id)
    if not row:
        connection.close()
        raise HTTPException(404, "experiment not found")

    context_rows = []
    for piece in str(row["context_ids"] or "").split(","):
        if piece.strip().isdigit():
            found = connection.execute(
                "select * from context_files where id = ?", (int(piece),)
            ).fetchone()
            if found:
                context_rows.append(found)
    connection.close()

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        if row["prompt_py"]:
            bundle.writestr("prompt.py", row["prompt_py"])
        if row["mix_py"]:
            bundle.writestr("mix.py", row["mix_py"])
        elif row["mix_file"]:
            bundle.writestr("default_mix.py", DEFAULT_MIX_SOURCE)
        if row["speech_text"]:
            bundle.writestr("answer.txt", row["speech_text"])
        if row["run_log"]:
            bundle.writestr("log.txt", row["run_log"])
        if row["validation"]:
            bundle.writestr("validation.txt", row["validation"])
        if row["reflection"]:
            bundle.writestr("reflection.txt", row["reflection"])

        info = [
            f"experiment {row['id']}",
            f"title: {row['title'] or 'none'}",
            f"created {row['created_at']}",
            f"protocol: {row['protocol_id'] or row['id']}, day {row['day_number'] or 1}",
            f"continues from: {row['prior_id'] if row['prior_id'] else 'nothing, this is day 1'}",
            f"topic: {row['topic'] or 'none'}",
            f"model: {row['model'] or 'none'}",
            f"prompt taken from: {row['prompt_source'] or 'unknown'}",
            f"word count: {row['word_count'] or 0}",
            f"voice id: {row['voice_id'] or 'none'}",
            f"stability: {row['stability']}",
            f"style: {row['style']}",
            f"speaker boost: {'on' if row['boost'] else 'off'}",
            f"music: {row['music_filename'] or 'none'}",
            f"mix source: {row['mix_source'] or 'none'}",
            f"music gain db: {row['music_gain_db'] if row['music_gain_db'] is not None else 'mix default'}",
            f"voice lufs: {row['voice_lufs'] if row['voice_lufs'] is not None else 'mix default'}",
            f"sync mode: {row['sync_mode'] or 'mix default'}",
            f"mix profile: {row['mix_profile'] or 'mix default'}",
            f"tts provider: {row['tts_provider'] or 'elevenlabs'}",
            f"tag: {row['tag'] or 'none'}",
            f"verdict: {row['verdict'] or 'not judged'}",
            f"comment: {row['comment'] or 'none'}",
        ]
        if row["parent_id"]:
            info.append(f"remix of experiment {row['parent_id']}")
        bundle.writestr("info.txt", "\n".join(info) + "\n")

        if context_rows:
            listing = []
            for found in context_rows:
                listing.append(f"{found['name']}  kind {found['kind']}  {found['chars']} chars  {found['note']}")
                source = CONTEXT_DIR / (found["stored_name"] or "")
                if found["stored_name"] and source.exists():
                    bundle.write(source, f"context/{found['name']}")
            bundle.writestr("context/context.txt", "\n".join(listing) + "\n")

        if row["mix_file"] and (AUDIO_DIR / row["mix_file"]).exists():
            bundle.write(AUDIO_DIR / row["mix_file"], "mix.mp3")
        if row["voice_file"] and (AUDIO_DIR / row["voice_file"]).exists():
            bundle.write(AUDIO_DIR / row["voice_file"], "voice.wav")
        if row["music_file"] and (MUSIC_DIR / row["music_file"]).exists():
            bundle.write(MUSIC_DIR / row["music_file"], "music_" + (row["music_filename"] or "track"))

    buffer.seek(0)
    return Response(
        content=buffer.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=experiment_{experiment_id}.zip"},
    )


@app.get("/api/bench/audio/{filename}")
def get_audio(filename: str, request: Request):
    safe_name = Path(filename).name
    file_path = AUDIO_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(404, "file not found")
    media_type = "audio/mpeg" if safe_name.endswith(".mp3") else "audio/wav"
    data = file_path.read_bytes()
    total = len(data)

    # range support so players can seek instead of always starting at byte zero
    range_header = request.headers.get("range")
    if range_header:
        try:
            range_spec = range_header.strip()
            if range_spec.lower().startswith("bytes="):
                range_spec = range_spec[6:]
            parts = range_spec.split("-", 1)
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if len(parts) > 1 and parts[1] else total - 1
            end = min(end, total - 1)
            if start > end or start >= total:
                return Response(content=b"", status_code=416,
                                headers={"Content-Range": f"bytes */{total}"})
            chunk = data[start:end + 1]
            return Response(
                content=chunk,
                status_code=206,
                media_type=media_type,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{total}",
                    "Content-Length": str(len(chunk)),
                    "Accept-Ranges": "bytes",
                    "Content-Disposition": f"inline; filename={safe_name}",
                },
            )
        except (ValueError, IndexError):
            pass

    return Response(
        content=data,
        media_type=media_type,
        headers={
            "Content-Length": str(total),
            "Accept-Ranges": "bytes",
            "Content-Disposition": f"inline; filename={safe_name}",
        },
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "stimgen-v2"}


HTML_PATH = Path(__file__).resolve().parent / "bench.html"


@app.get("/")
def serve_ui():
    if HTML_PATH.exists():
        return FileResponse(str(HTML_PATH), media_type="text/html", headers={"Cache-Control": "no-store"})
    return JSONResponse({"error": "bench.html not found"}, status_code=404)
