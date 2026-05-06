"""
Tests for the voice-to-voice conversion service and route layer.

These tests exercise the service helpers (record creation, status update,
list, delete) and the route layer using a FastAPI TestClient backed by an
in-memory SQLite database. They do NOT invoke actual model inference
(Whisper or TTS) — the S2S backend is replaced by a lightweight stub.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Point config at a temporary data directory for the test session."""
    from backend import config as backend_config

    original = backend_config._data_dir
    backend_config.set_data_dir(str(tmp_path))
    yield tmp_path
    backend_config._data_dir = original


@pytest.fixture
def db_session(tmp_data_dir):
    """In-memory SQLite session with all ORM tables created."""
    from backend.database.models import Base
    from sqlalchemy.pool import StaticPool

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def sample_profile(db_session):
    """Create a minimal voice profile for tests that need one."""
    from datetime import datetime

    from backend.database.models import VoiceProfile as DBVoiceProfile

    profile = DBVoiceProfile(
        id=str(uuid.uuid4()),
        name="Test Voice",
        language="en",
        voice_type="cloned",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db_session.add(profile)
    db_session.commit()
    return profile


# ---------------------------------------------------------------------------
# Service-layer tests
# ---------------------------------------------------------------------------


class TestVoiceConvertService:
    def test_create_and_get_conversion_record(self, db_session, sample_profile, tmp_data_dir):
        from backend.services.voice_convert import create_conversion_record, get_conversion

        cid = str(uuid.uuid4())
        src = tmp_data_dir / "voice_conversions" / f"{cid}_source.wav"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

        row = create_conversion_record(
            conversion_id=cid,
            profile_id=sample_profile.id,
            source_audio_path=str(src),
            engine="qwen",
            model_size="1.7B",
            language="en",
            seed=None,
            db=db_session,
        )
        assert row.id == cid
        assert row.status == "generating"

        fetched = get_conversion(cid, db_session)
        assert fetched is not None
        assert fetched.id == cid

    def test_list_conversions_returns_all(self, db_session, sample_profile, tmp_data_dir):
        from backend.services.voice_convert import create_conversion_record, list_conversions

        for _ in range(3):
            cid = str(uuid.uuid4())
            src = tmp_data_dir / "voice_conversions" / f"{cid}_source.wav"
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_bytes(b"")
            create_conversion_record(
                conversion_id=cid,
                profile_id=sample_profile.id,
                source_audio_path=str(src),
                engine="qwen",
                model_size="1.7B",
                language="en",
                seed=None,
                db=db_session,
            )

        items, total = list_conversions(db_session)
        assert total == 3
        assert len(items) == 3

    def test_delete_conversion(self, db_session, sample_profile, tmp_data_dir):
        from backend.services.voice_convert import (
            create_conversion_record,
            delete_conversion,
            get_conversion,
        )

        cid = str(uuid.uuid4())
        src = tmp_data_dir / "voice_conversions" / f"{cid}_source.wav"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(b"dummy")

        create_conversion_record(
            conversion_id=cid,
            profile_id=sample_profile.id,
            source_audio_path=str(src),
            engine="qwen",
            model_size="1.7B",
            language="en",
            seed=None,
            db=db_session,
        )

        deleted = delete_conversion(cid, db_session)
        assert deleted is True
        assert get_conversion(cid, db_session) is None

    def test_delete_nonexistent_returns_false(self, db_session):
        from backend.services.voice_convert import delete_conversion

        assert delete_conversion("nonexistent-id", db_session) is False


# ---------------------------------------------------------------------------
# Route-layer tests (using a stub S2S backend)
# ---------------------------------------------------------------------------


class _StubS2SBackend:
    """Fake S2SBackend that avoids loading any real model."""

    async def convert(
        self,
        source_audio_path: str,
        profile_id: str,
        db,
        engine: str = "qwen",
        model_size: str = "1.7B",
        language: str = "en",
        seed=None,
        stt_model: str = "turbo",
    ) -> Tuple[np.ndarray, int, str]:
        audio = np.zeros(24000, dtype=np.float32)
        return audio, 24000, "hello world"


@pytest.fixture
def app_client(tmp_data_dir, monkeypatch):
    """Build a TestClient wired to an in-memory DB and a stub S2S backend."""
    from backend.database.models import Base
    from backend.database.session import get_db

    # -- in-memory database --------------------------------------------------
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    def override_get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    # -- stub S2S backend ----------------------------------------------------
    stub_s2s = _StubS2SBackend()

    import backend.backends as backends_mod

    monkeypatch.setattr(backends_mod, "get_s2s_backend", lambda: stub_s2s)

    # -- stub engine validation -----------------------------------------------
    monkeypatch.setattr(backends_mod, "engine_has_model_sizes", lambda e: e in ("qwen", "qwen_custom_voice"))

    # -- stub task queue (no asyncio event loop needed) ----------------------
    import backend.services.task_queue as tq_mod
    import asyncio

    monkeypatch.setattr(tq_mod, "enqueue_generation", lambda _id, _coro: None)

    # -- build a minimal FastAPI app with just the voice-convert router ------
    from fastapi import FastAPI
    from backend.routes.voice_convert import router as vc_router

    app = FastAPI()
    app.include_router(vc_router)
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app, raise_server_exceptions=False) as client:
        # Seed a profile so the route can find it
        db = Session()
        from datetime import datetime

        from backend.database.models import VoiceProfile as DBVoiceProfile

        profile = DBVoiceProfile(
            id="test-profile-id",
            name="Test Voice",
            language="en",
            voice_type="cloned",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(profile)
        db.commit()
        db.close()

        yield client, "test-profile-id"


class TestVoiceConvertRoutes:
    def _make_wav_bytes(self) -> bytes:
        """Return a minimal valid WAV file (silence)."""
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, np.zeros(24000, dtype=np.float32), 24000, format="WAV")
        buf.seek(0)
        return buf.read()

    def test_list_empty(self, app_client):
        client, _ = app_client
        resp = client.get("/voice-convert")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []

    def test_start_convert_missing_profile(self, app_client):
        client, _ = app_client
        wav = self._make_wav_bytes()
        resp = client.post(
            "/voice-convert",
            data={"profile_id": "nonexistent", "engine": "qwen"},
            files={"file": ("test.wav", wav, "audio/wav")},
        )
        assert resp.status_code == 404

    def test_start_convert_bad_extension(self, app_client):
        client, profile_id = app_client
        resp = client.post(
            "/voice-convert",
            data={"profile_id": profile_id, "engine": "qwen"},
            files={"file": ("test.txt", b"not audio", "text/plain")},
        )
        assert resp.status_code == 400

    def test_get_nonexistent_conversion(self, app_client):
        client, _ = app_client
        resp = client.get("/voice-convert/nonexistent-id")
        assert resp.status_code == 404

    def test_delete_nonexistent(self, app_client):
        client, _ = app_client
        resp = client.delete("/voice-convert/nonexistent-id")
        assert resp.status_code == 404
