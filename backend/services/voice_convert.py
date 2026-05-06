"""
Voice-to-Voice conversion service.

Orchestrates the LocalVCBackend (or any future S2SBackend) inside the
existing serial task queue so GPU / model inference does not race with TTS
generation.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from .. import config
from ..database import VoiceConversion as DBVoiceConversion, get_db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def _update_status(
    conversion_id: str,
    status: str,
    db: Session,
    *,
    audio_path: Optional[str] = None,
    duration: Optional[float] = None,
    transcript: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    row = db.query(DBVoiceConversion).filter_by(id=conversion_id).first()
    if row is None:
        return
    row.status = status
    if audio_path is not None:
        row.audio_path = audio_path
    if duration is not None:
        row.duration = duration
    if transcript is not None:
        row.transcript = transcript
    if error is not None:
        row.error = error
    db.commit()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_voice_convert(
    *,
    conversion_id: str,
    source_audio_path: str,
    profile_id: str,
    engine: str,
    model_size: str,
    language: str,
    seed: Optional[int],
    stt_model: str,
) -> None:
    """Execute the voice-conversion pipeline and persist the result.

    Designed to be enqueued via the existing serial task queue so it cannot
    race with TTS inference on the same GPU.
    """
    from ..backends import get_s2s_backend
    from ..utils.audio import normalize_audio, save_audio

    bg_db = next(get_db())
    try:
        _update_status(conversion_id, "generating", bg_db)

        s2s = get_s2s_backend()
        audio, sample_rate, transcript = await s2s.convert(
            source_audio_path=source_audio_path,
            profile_id=profile_id,
            db=bg_db,
            engine=engine,
            model_size=model_size,
            language=language,
            seed=seed,
            stt_model=stt_model,
        )

        audio = normalize_audio(audio)
        duration = len(audio) / sample_rate

        out_path = config.get_voice_conversions_dir() / f"{conversion_id}.wav"
        save_audio(audio, str(out_path), sample_rate)

        _update_status(
            conversion_id,
            "completed",
            bg_db,
            audio_path=config.to_storage_path(out_path),
            duration=duration,
            transcript=transcript,
        )
        logger.info("VoiceConvert %s completed (%.1fs)", conversion_id, duration)

    except asyncio.CancelledError:
        _update_status(conversion_id, "failed", bg_db, error="Conversion cancelled")
        raise
    except Exception as exc:
        traceback.print_exc()
        _update_status(conversion_id, "failed", bg_db, error=str(exc))
    finally:
        bg_db.close()


def create_conversion_record(
    *,
    conversion_id: str,
    profile_id: str,
    source_audio_path: str,
    engine: str,
    model_size: Optional[str],
    language: str,
    seed: Optional[int],
    db: Session,
) -> DBVoiceConversion:
    """Insert a new voice_conversions row and return the ORM object."""
    row = DBVoiceConversion(
        id=conversion_id,
        profile_id=profile_id,
        source_audio_path=config.to_storage_path(source_audio_path),
        engine=engine,
        model_size=model_size,
        language=language,
        seed=seed,
        status="generating",
        created_at=datetime.utcnow(),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_conversion(conversion_id: str, db: Session) -> Optional[DBVoiceConversion]:
    return db.query(DBVoiceConversion).filter_by(id=conversion_id).first()


def list_conversions(
    db: Session,
    profile_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[DBVoiceConversion], int]:
    q = db.query(DBVoiceConversion)
    if profile_id:
        q = q.filter(DBVoiceConversion.profile_id == profile_id)
    total = q.count()
    items = q.order_by(DBVoiceConversion.created_at.desc()).offset(offset).limit(limit).all()
    return items, total


def delete_conversion(conversion_id: str, db: Session) -> bool:
    row = db.query(DBVoiceConversion).filter_by(id=conversion_id).first()
    if not row:
        return False

    # Remove audio file
    if row.audio_path:
        p = config.resolve_storage_path(row.audio_path)
        if p and p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    # Remove source audio file stored in the voice_conversions dir
    if row.source_audio_path:
        sp = config.resolve_storage_path(row.source_audio_path)
        if sp and sp.exists():
            try:
                sp.unlink()
            except OSError:
                pass

    db.delete(row)
    db.commit()
    return True
