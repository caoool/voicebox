"""Voice-to-voice conversion endpoints."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .. import config, models
from ..database import VoiceConversion as DBVoiceConversion, VoiceProfile as DBVoiceProfile, get_db
from ..models import VoiceConversionResponse, VoiceConversionListResponse
from ..services import voice_convert as vc_service
from ..services.task_queue import enqueue_generation

logger = logging.getLogger(__name__)

router = APIRouter()

_ALLOWED_AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}
_MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB


@router.post("/voice-convert", response_model=VoiceConversionResponse)
async def start_voice_convert(
    file: UploadFile = File(..., description="Source audio to convert"),
    profile_id: str = Form(...),
    engine: str = Form(default="qwen"),
    model_size: str = Form(default="1.7B"),
    language: str = Form(default="en"),
    seed: int | None = Form(default=None),
    stt_model: str = Form(default="turbo"),
    db: Session = Depends(get_db),
):
    """Upload source audio and start a voice-to-voice conversion.

    The conversion is enqueued in the serial inference queue (same queue as
    TTS generation) to prevent GPU contention.  Poll ``GET
    /voice-convert/{id}/status`` for progress.
    """
    # Validate profile
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    # Validate file extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_AUDIO_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{suffix}'. Allowed: {sorted(_ALLOWED_AUDIO_EXT)}",
        )

    # Read uploaded bytes (with size guard)
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
            )
        chunks.append(chunk)
    audio_bytes = b"".join(chunks)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    # Save source audio to voice_conversions dir
    conversion_id = str(uuid.uuid4())
    source_path = config.get_voice_conversions_dir() / f"{conversion_id}_source{suffix}"
    source_path.write_bytes(audio_bytes)

    # Validate the audio is actually decodable
    try:
        from ..utils.audio import load_audio

        load_audio(str(source_path))
    except Exception as decode_err:
        try:
            source_path.unlink()
        except OSError:
            pass
        raise HTTPException(
            status_code=400, detail=f"Could not decode audio: {decode_err}"
        ) from decode_err

    # Resolve effective model_size
    from ..backends import engine_has_model_sizes

    effective_model_size = model_size if engine_has_model_sizes(engine) else None

    # Create DB record
    row = vc_service.create_conversion_record(
        conversion_id=conversion_id,
        profile_id=profile_id,
        source_audio_path=str(source_path),
        engine=engine,
        model_size=effective_model_size,
        language=language,
        seed=seed,
        db=db,
    )

    # Enqueue in the shared serial inference queue
    enqueue_generation(
        conversion_id,
        vc_service.run_voice_convert(
            conversion_id=conversion_id,
            source_audio_path=str(source_path),
            profile_id=profile_id,
            engine=engine,
            model_size=effective_model_size or "1.7B",
            language=language,
            seed=seed,
            stt_model=stt_model,
        ),
    )

    return VoiceConversionResponse.model_validate(row)


@router.get("/voice-convert", response_model=VoiceConversionListResponse)
def list_voice_converts(
    profile_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """List voice conversion records (newest first)."""
    items, total = vc_service.list_conversions(db, profile_id=profile_id, limit=limit, offset=offset)
    return VoiceConversionListResponse(
        items=[VoiceConversionResponse.model_validate(r) for r in items],
        total=total,
    )


@router.get("/voice-convert/{conversion_id}", response_model=VoiceConversionResponse)
def get_voice_convert(conversion_id: str, db: Session = Depends(get_db)):
    """Get a voice conversion record by ID."""
    row = vc_service.get_conversion(conversion_id, db)
    if not row:
        raise HTTPException(status_code=404, detail="Voice conversion not found")
    return VoiceConversionResponse.model_validate(row)


@router.get("/voice-convert/{conversion_id}/status")
async def stream_voice_convert_status(conversion_id: str, db: Session = Depends(get_db)):
    """SSE endpoint that streams conversion status updates."""
    import json

    async def event_stream():
        try:
            while True:
                db.expire_all()
                row = db.query(DBVoiceConversion).filter_by(id=conversion_id).first()
                if not row:
                    yield f"data: {json.dumps({'status': 'not_found', 'id': conversion_id})}\n\n"
                    return

                payload = {
                    "id": row.id,
                    "status": row.status or "generating",
                    "duration": row.duration,
                    "error": row.error,
                    "transcript": row.transcript,
                    "audio_path": row.audio_path,
                }
                yield f"data: {json.dumps(payload)}\n\n"

                if (row.status or "generating") in ("completed", "failed"):
                    return

                await asyncio.sleep(1)
        except (BrokenPipeError, ConnectionResetError, asyncio.CancelledError):
            logger.debug("SSE client disconnected for conversion %s", conversion_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/voice-convert/{conversion_id}/cancel")
def cancel_voice_convert(conversion_id: str, db: Session = Depends(get_db)):
    """Cancel a queued or running conversion."""
    from ..services.task_queue import cancel_generation as cancel_job

    row = vc_service.get_conversion(conversion_id, db)
    if not row:
        raise HTTPException(status_code=404, detail="Voice conversion not found")

    if (row.status or "generating") not in ("generating",):
        raise HTTPException(status_code=400, detail="Only active conversions can be cancelled")

    cancel_job(conversion_id)

    row = db.query(DBVoiceConversion).filter_by(id=conversion_id).first()
    if row:
        row.status = "failed"
        row.error = "Cancelled by user"
        db.commit()

    return {"message": "Conversion cancellation requested"}


@router.delete("/voice-convert/{conversion_id}")
def delete_voice_convert(conversion_id: str, db: Session = Depends(get_db)):
    """Delete a voice conversion record and its associated files."""
    deleted = vc_service.delete_conversion(conversion_id, db)
    if not deleted:
        raise HTTPException(status_code=404, detail="Voice conversion not found")
    return {"message": "Deleted"}
