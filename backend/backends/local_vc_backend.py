"""
Local Voice-Conversion Backend (STT → TTS pipeline).

This is the first-version S2SBackend implementation.  Rather than requiring
a dedicated voice-conversion model, it chains the existing STT (Whisper) and
TTS engines to produce a reasonable approximation:

    1. Transcribe ``source_audio`` with Whisper.
    2. Synthesise the transcript with the target voice profile via the
       configured TTS engine.

This approach reuses all existing model infrastructure, requires no new model
downloads, and can be tested end-to-end immediately.  A future backend can
swap in a dedicated end-to-end VC model (e.g. OpenVoice, RVC, KNNVC) without
changing the service or route layers — they will keep calling ``convert()``.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LocalVCBackend:
    """STT → TTS pipeline voice conversion backend.

    Satisfies the ``S2SBackend`` protocol defined in ``backends/__init__.py``.
    """

    async def convert(
        self,
        source_audio_path: str,
        profile_id: str,
        db,
        engine: str = "qwen",
        model_size: str = "1.7B",
        language: str = "en",
        seed: Optional[int] = None,
        stt_model: str = "turbo",
    ) -> Tuple[np.ndarray, int, str]:
        """Convert source audio to the target speaker's voice.

        Steps:
        1. Transcribe ``source_audio_path`` with Whisper (``stt_model``).
        2. Load target voice profile.
        3. Synthesise the transcript with the TTS engine.

        Returns:
            Tuple of (audio_array, sample_rate, transcript)
        """
        from . import (
            get_tts_backend_for_engine,
            get_stt_backend,
            load_engine_model,
            engine_needs_trim,
        )
        from ..services import profiles as profiles_service
        from ..utils.chunked_tts import generate_chunked
        from ..utils.audio import trim_tts_output

        # ── Step 1: transcribe source audio ──────────────────────────────────
        logger.info("VoiceConvert: transcribing %s with Whisper %s", source_audio_path, stt_model)
        stt_model_instance = get_stt_backend()
        transcript = await stt_model_instance.transcribe(
            source_audio_path,
            language=language if language != "auto" else None,
            model_size=stt_model,
        )
        transcript = (transcript or "").strip()
        if not transcript:
            raise ValueError(
                "Source audio produced an empty transcript — try a clearer recording or a different Whisper model."
            )
        logger.info("VoiceConvert: transcript = %r", transcript[:120])

        # ── Step 2: load target TTS engine ────────────────────────────────────
        logger.info("VoiceConvert: loading TTS engine %s / %s", engine, model_size)
        await load_engine_model(engine, model_size)

        tts_model = get_tts_backend_for_engine(engine)
        voice_prompt = await profiles_service.create_voice_prompt_for_profile(
            profile_id,
            db,
            use_cache=True,
            engine=engine,
        )

        # ── Step 3: synthesise transcript ─────────────────────────────────────
        logger.info("VoiceConvert: synthesising with engine %s", engine)
        trim_fn = trim_tts_output if engine_needs_trim(engine) else None
        audio, sample_rate = await generate_chunked(
            tts_model,
            transcript,
            voice_prompt,
            language=language,
            seed=seed,
            trim_fn=trim_fn,
        )
        return audio, sample_rate, transcript
