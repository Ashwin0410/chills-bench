import inspect
import io
import json
import os
import re
import sqlite3
import sys
import time
import types
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
DB_PATH = DATA_DIR / "bench.db"

for folder in (DATA_DIR, AUDIO_DIR, MUSIC_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Chills Bench")


# database

def db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


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
    try:
        connection.execute("alter table experiments add column mix_source text default ''")
    except Exception:
        pass
    connection.commit()
    connection.close()


init_db()


# audio helpers, ported from the main repo so pasted mix.py files work standalone

def load_audio(path):
    return AudioSegment.from_file(path)


def normalize_dbfs(segment, target_dbfs):
    return segment.apply_gain(target_dbfs - segment.dBFS)


def make_stereo(segment):
    return segment.set_channels(2)


def duration_ms(segment):
    return len(segment)


def content_duration_sec(music_path):
    # same measurement as the repo's audio.py, scans back from the end for where sound stops
    import math
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


# exec loaders for pasted files

def exec_pasted(code, module_name):
    namespace = {
        "__name__": module_name,
        "__package__": "app.services",
        "__builtins__": __builtins__,
    }
    exec(compile(code, module_name + ".py", "exec"), namespace)
    return namespace


def load_prompt_file(code, topic):
    # returns (system_prompt, user_prompt) built from the pasted file plus the topic
    try:
        namespace = exec_pasted(code, "pasted_prompt")
    except Exception as error:
        raise HTTPException(400, f"prompt file failed to run: {error}")

    system_prompt = None
    for name in ("SYSTEM_PROMPT", "MEDITATION_SYSTEM", "SPEECH1_SYSTEM", "JOURNAL_SYSTEM", "PLAN_SYSTEM"):
        value = namespace.get(name)
        if isinstance(value, str) and value.strip():
            system_prompt = value
            break
    if system_prompt is None:
        # fall back to the largest string in the file
        strings = [v for v in namespace.values() if isinstance(v, str) and len(v) > 200]
        if strings:
            system_prompt = max(strings, key=len)
    if system_prompt is None:
        raise HTTPException(400, "no SYSTEM_PROMPT or usable prompt string found in the pasted file")

    builder = namespace.get("build_user_prompt")
    if callable(builder):
        try:
            user_prompt = call_builder(builder, topic)
        except Exception as error:
            raise HTTPException(400, f"build_user_prompt failed: {error}")
    else:
        user_prompt = topic if topic else "Write the piece now."
    return system_prompt, user_prompt


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


def default_mix(voice_path, music_path, out_path, **_):
    # plain fallback used when the mix box is empty
    voice = make_stereo(load_audio(voice_path).set_frame_rate(44100))
    if music_path and Path(music_path).exists():
        music = make_stereo(load_audio(music_path).set_frame_rate(44100))
        music = music.apply_gain(-14.0)
        if len(music) < len(voice):
            loops = len(voice) // len(music) + 1
            music = music * loops
        music = music[:len(voice) + 3000].fade_out(2000)
        mixed = music.overlay(voice)
    else:
        mixed = voice
    mixed.export(out_path, format="mp3", bitrate="256k")
    return len(mixed)


def mix_source_for(mix_py, music_path):
    if mix_py.strip():
        return "pasted mix.py"
    if music_path:
        return "bench default"
    return "voice only"


def run_mix(mix_function, voice_path, music_path, out_path):
    content_sec = None
    if music_path and Path(str(music_path)).exists():
        content_sec = content_duration_sec(music_path)
    try:
        try:
            if content_sec is not None:
                mix_function(voice_path=str(voice_path), music_path=str(music_path),
                             out_path=str(out_path), content_duration_sec=content_sec)
            else:
                mix_function(voice_path=str(voice_path), music_path=str(music_path), out_path=str(out_path))
        except TypeError:
            # pasted mix does not accept content_duration_sec or keywords, fall back
            try:
                mix_function(voice_path=str(voice_path), music_path=str(music_path), out_path=str(out_path))
            except TypeError:
                mix_function(str(voice_path), str(music_path), str(out_path))
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(500, f"mix failed: {error}")
    if not Path(out_path).exists():
        raise HTTPException(500, "mix ran but produced no output file")


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
PAUSE_SENTINEL = "<<<PAUSE>>>"
MAX_CHARS = 3200
PAUSE_MS = 900
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


def synth_chunk(text, voice_id, voice_settings):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "accept": "audio/mpeg", "Content-Type": "application/json"}
    payload = {"text": text, "model_id": "eleven_v3", "voice_settings": voice_settings}
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


def synth(text, voice_id, voice_settings, out_path):
    if not ELEVENLABS_API_KEY:
        raise HTTPException(503, "ELEVENLABS_API_KEY is not set on the server")
    raw = text.strip().replace("[breath]", " ").replace(PAUSE_TOKEN, f" {PAUSE_SENTINEL} ")
    blocks = [block.strip() for block in raw.split(PAUSE_SENTINEL) if block.strip()]
    if not blocks:
        raise HTTPException(400, "speech text is empty after cleanup")

    segments = []
    for block_index, block in enumerate(blocks):
        for part_index, part in enumerate(split_chunks(block)):
            print(f"tts block {block_index + 1}/{len(blocks)} chunk {part_index + 1}, {len(part)} chars")
            segments.append(synth_chunk(part, voice_id, voice_settings))
            segments.append(AudioSegment.silent(duration=CHUNK_GAP_MS, frame_rate=44100))
        segments.pop()
        if block_index < len(blocks) - 1:
            segments.append(AudioSegment.silent(duration=PAUSE_MS, frame_rate=44100))

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
    except Exception as error:
        print(f"noise reduction skipped: {error}")

    full.export(out_path, format="wav")
    return out_path


# experiment helpers

def now():
    return datetime.now(timezone.utc).isoformat()


def experiment_row(row):
    return {
        "id": row["id"],
        "created_at": row["created_at"],
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
        "verdict": row["verdict"],
        "comment": row["comment"],
        "parent_id": row["parent_id"],
    }


def save_upload(upload, destination):
    with open(destination, "wb") as handle:
        while True:
            piece = upload.file.read(1024 * 1024)
            if not piece:
                break
            handle.write(piece)


def fetch_experiment(connection, experiment_id):
    return connection.execute("select * from experiments where id = ?", (experiment_id,)).fetchone()


# endpoints

class WriteReq(BaseModel):
    topic: str = ""
    prompt_py: str = ""
    model: str = "claude-sonnet-4-6"


@app.post("/api/bench/write")
def write_answer(req: WriteReq):
    if not req.prompt_py.strip():
        raise HTTPException(400, "paste a prompt file first")
    system_prompt, user_prompt = load_prompt_file(req.prompt_py, req.topic.strip())
    speech = call_claude(req.model, system_prompt, user_prompt)
    if not speech:
        raise HTTPException(502, "claude returned empty text")
    # save on write: every generation becomes a card, audio empty until make runs
    connection = db()
    cursor = connection.execute(
        "insert into experiments (created_at, topic, prompt_py, model, speech_text) values (?, ?, ?, ?, ?)",
        (now(), req.topic.strip(), req.prompt_py, req.model, speech),
    )
    experiment_id = cursor.lastrowid
    connection.commit()
    connection.close()
    print(f"experiment {experiment_id} text saved")
    return {"speech": speech, "experiment_id": experiment_id}


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
    connection.commit()
    return cursor.lastrowid, False


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
    experiment_id: int = Form(0),
    voice_only: bool = Form(False),
    music: UploadFile | None = File(default=None),
):
    speech = speech.strip()
    voice_id = voice_id.strip()
    if not speech:
        raise HTTPException(400, "speech text is empty")
    if not voice_id:
        raise HTTPException(400, "voice id is empty")

    if voice_only:
        mix_py = ""
        music = None

    mix_function = load_mix_function(mix_py)

    connection = db()
    target_id, attached = attach_or_create(connection, experiment_id, topic.strip(), prompt_py, model, speech)

    try:
        music_path = ""
        music_filename = ""
        if music is not None and music.filename:
            music_filename = Path(music.filename).name
            music_path = MUSIC_DIR / f"{target_id}_{music_filename}"
            save_upload(music, music_path)

        source = mix_source_for(mix_py, music_path)

        voice_settings = {"stability": stability, "similarity_boost": 0.7, "style": style, "use_speaker_boost": boost}
        voice_file = f"{target_id}_voice.wav"
        synth(speech, voice_id, voice_settings, str(AUDIO_DIR / voice_file))
        print(f"experiment {target_id} voice saved")

        if voice_only:
            mix_file = ""
        else:
            mix_file = f"{target_id}_mix.mp3"
            run_mix(mix_function, AUDIO_DIR / voice_file, music_path, AUDIO_DIR / mix_file)
            print(f"experiment {target_id} mix saved, {source}")

        connection.execute(
            "update experiments set topic = ?, prompt_py = ?, mix_py = ?, model = ?, voice_id = ?, "
            "stability = ?, style = ?, boost = ?, speech_text = ?, music_filename = ?, music_file = ?, "
            "voice_file = ?, mix_file = ?, mix_source = ? where id = ?",
            (topic.strip(), prompt_py, mix_py, model, voice_id, stability, style, int(boost), speech,
             music_filename, str(Path(music_path).name) if music_path else "",
             voice_file, mix_file, source, target_id),
        )
        connection.commit()
        row = fetch_experiment(connection, target_id)
        return experiment_row(row)
    except HTTPException:
        if not attached:
            connection.execute("delete from experiments where id = ?", (target_id,))
            connection.commit()
        raise
    except Exception as error:
        if not attached:
            connection.execute("delete from experiments where id = ?", (target_id,))
            connection.commit()
        raise HTTPException(500, f"make failed: {error}")
    finally:
        connection.close()


@app.post("/api/bench/remix/{experiment_id}")
def remix(
    experiment_id: int,
    mix_py: str = Form(""),
    music: UploadFile | None = File(default=None),
):
    connection = db()
    parent = fetch_experiment(connection, experiment_id)
    if not parent:
        connection.close()
        raise HTTPException(404, "experiment not found")
    if not parent["voice_file"] or not (AUDIO_DIR / parent["voice_file"]).exists():
        connection.close()
        raise HTTPException(400, "no stored voice file for this experiment")

    mix_function = load_mix_function(mix_py)

    music_path = ""
    music_filename = parent["music_filename"]
    if music is not None and music.filename:
        music_filename = Path(music.filename).name
    elif parent["music_file"] and (MUSIC_DIR / parent["music_file"]).exists():
        music_path = MUSIC_DIR / parent["music_file"]

    cursor = connection.execute(
        "insert into experiments (created_at, topic, prompt_py, mix_py, model, voice_id, stability, style, boost, speech_text, voice_file, parent_id) "
        "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (now(), parent["topic"], parent["prompt_py"], mix_py, parent["model"], parent["voice_id"],
         parent["stability"], parent["style"], parent["boost"], parent["speech_text"],
         parent["voice_file"], experiment_id),
    )
    new_id = cursor.lastrowid
    connection.commit()

    try:
        if music is not None and music.filename:
            music_path = MUSIC_DIR / f"{new_id}_{music_filename}"
            save_upload(music, music_path)

        source = mix_source_for(mix_py, music_path)

        mix_file = f"{new_id}_mix.mp3"
        run_mix(mix_function, AUDIO_DIR / parent["voice_file"], music_path, AUDIO_DIR / mix_file)
        print(f"experiment {new_id} remix of {experiment_id} saved, {source}")

        connection.execute(
            "update experiments set music_filename = ?, music_file = ?, mix_file = ?, mix_source = ? where id = ?",
            (music_filename, str(Path(music_path).name) if music_path else "", mix_file, source, new_id),
        )
        connection.commit()
        row = fetch_experiment(connection, new_id)
        return experiment_row(row)
    except HTTPException:
        connection.execute("delete from experiments where id = ?", (new_id,))
        connection.commit()
        raise
    except Exception as error:
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

    for field, folder in (("voice_file", AUDIO_DIR), ("mix_file", AUDIO_DIR), ("music_file", MUSIC_DIR)):
        name = row[field]
        if name:
            path = folder / name
            if path.exists():
                path.unlink()

    connection.execute("delete from experiments where id = ?", (experiment_id,))
    connection.commit()
    renumber_experiments(connection)
    connection.close()
    print(f"deleted experiment {experiment_id} and renumbered")
    return {"status": "ok"}


def renumber_experiments(connection):
    # close the gap after a delete: ids become 1..n in creation order,
    # audio and music files renamed to match, remix parents remapped,
    # parents that no longer exist become null
    rows = connection.execute("select * from experiments order by id").fetchall()
    id_map = {}
    for position, row in enumerate(rows, start=1):
        id_map[row["id"]] = position

    for row in rows:
        old_id = row["id"]
        new_id = id_map[old_id]
        if new_id == old_id:
            continue

        renames = []
        if row["voice_file"]:
            renames.append((AUDIO_DIR, row["voice_file"], f"{new_id}_voice.wav", "voice_file"))
        if row["mix_file"]:
            renames.append((AUDIO_DIR, row["mix_file"], f"{new_id}_mix.mp3", "mix_file"))
        if row["music_file"]:
            tail = row["music_file"].split("_", 1)[1] if "_" in row["music_file"] else row["music_file"]
            renames.append((MUSIC_DIR, row["music_file"], f"{new_id}_{tail}", "music_file"))

        updates = {}
        for folder, old_name, new_name, field in renames:
            old_path = folder / old_name
            if old_path.exists():
                old_path.rename(folder / new_name)
            updates[field] = new_name

        new_parent = None
        if row["parent_id"]:
            new_parent = id_map.get(row["parent_id"])

        connection.execute(
            "update experiments set id = ?, voice_file = ?, mix_file = ?, music_file = ?, parent_id = ? where id = ?",
            (new_id,
             updates.get("voice_file", row["voice_file"]),
             updates.get("mix_file", row["mix_file"]),
             updates.get("music_file", row["music_file"]),
             new_parent,
             old_id),
        )

    # parents pointing at deleted experiments become null even when ids did not move
    surviving = set(id_map.values())
    for row in connection.execute("select id, parent_id from experiments where parent_id is not null").fetchall():
        if row["parent_id"] not in surviving:
            connection.execute("update experiments set parent_id = null where id = ?", (row["id"],))

    total = len(rows)
    connection.execute("delete from sqlite_sequence where name = 'experiments'")
    if total:
        connection.execute("insert into sqlite_sequence (name, seq) values ('experiments', ?)", (total,))
    connection.commit()



@app.get("/api/bench/experiments")
def list_experiments():
    connection = db()
    rows = connection.execute("select * from experiments order by id desc").fetchall()
    connection.close()
    return {"experiments": [experiment_row(row) for row in rows]}


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
    return {"status": "ok", "service": "chills-bench"}


HTML_PATH = Path(__file__).resolve().parent / "bench.html"


@app.get("/")
def serve_ui():
    if HTML_PATH.exists():
        return FileResponse(str(HTML_PATH), media_type="text/html", headers={"Cache-Control": "no-store"})
    return JSONResponse({"error": "bench.html not found"}, status_code=404)