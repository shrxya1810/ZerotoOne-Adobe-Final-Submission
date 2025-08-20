"""
Podcast router for generating audio summaries and scripts from document content.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Header
from fastapi.responses import FileResponse
import logging
import os
import tempfile
from typing import Dict, List, Optional
import json

from ..settings import settings
from ..services.storage import storage_service
from ..services.index import semantic_search
from ..services.tts import tts_service
from ..models.schemas import (
    PodcastGenerationRequest, PodcastGenerationResponse,
    PodcastScriptRequest, PodcastScriptResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()


async def generate_podcast_script(session_id: str, topic: str, style: str = "conversational", max_sections: int = 5, current_pdf: Optional[str] = None) -> Dict:
    """
    Generate a podcast script from session documents using AI.

    Args:
        session_id: Session to analyze
        topic: Main topic for the podcast
        style: Podcast style (conversational, educational, professional)
        max_sections: Maximum number of document sections to include
        current_pdf: The currently opened PDF file to prioritize

    Returns:
        Generated script with speaker segments
    """
    if not settings.GEMINI_API_KEY:
        return {
            "script": "AI script generation not available. Please configure GEMINI_API_KEY.",
            "segments": [],
            "estimated_duration": "0:00"
        }

    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL)

        # Get relevant content
        relevant_chunks = semantic_search(session_id, topic, max_sections * 2)

        # Prioritize content from the current PDF
        if current_pdf:
            relevant_chunks = [chunk for chunk in relevant_chunks if chunk.get('document') == current_pdf] + \
                              [chunk for chunk in relevant_chunks if chunk.get('document') != current_pdf]

        if not relevant_chunks:
            return {
                "script": "No relevant content found for the specified topic.",
                "segments": [],
                "estimated_duration": "0:00"
            }

        # Prepare content summary (limit to ~1200 chars for 1-2 min podcast)
        max_script_chars = 1500
        content_summary = ""
        char_count = 0
        for i, chunk in enumerate(relevant_chunks[:max_sections]):
            section = f"\n--- Section {i+1} (from {chunk.get('document', 'Unknown')}) ---\n"
            section_content = chunk.get('content', '')[:400] + "...\n"
            if char_count + len(section) + len(section_content) > max_script_chars:
                break
            content_summary += section + section_content
            char_count += len(section) + len(section_content)

        # Style-specific prompts
        style_prompts = {
            "conversational": "Create a natural, conversational podcast script with two hosts discussing the content. Use casual language and include natural transitions. The total script should be no more than 1200 characters (about 1-2 minutes of speech).",
            "educational": "Create an educational podcast script with clear explanations and examples. Structure it as a teacher explaining concepts to students. The total script should be no more than 1200 characters (about 1-2 minutes of speech).",
            "professional": "Create a professional podcast script suitable for business audiences. Use formal language and focus on actionable insights. The total script should be no more than 1200 characters (about 1-2 minutes of speech)."
        }

        prompt = f"""
        You are creating a podcast script about \"{topic}\" based on the following content.

        Style: {style_prompts.get(style, style_prompts['conversational'])}

        Content to base the script on:
        {content_summary}

        Create a podcast script with:
        1. An engaging introduction to the topic
        2. 2-3 main discussion points based on the content
        3. Natural transitions between topics
        4. A conclusion with key takeaways

        Format as JSON with:
        - "script": full script text (no more than 1200 characters)
        - "segments": array of objects with "speaker", "text", and "duration_seconds"
        - "estimated_duration": total estimated time as "MM:SS"

        Each segment should be 20-60 seconds of speaking time.
        Include speaker names like "Host" or "Alex" and "Jordan" for conversational style.
        """

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )

        result = json.loads(response.text)
        # Truncate script to 1500 chars if needed (extra safety)
        if 'script' in result and len(result['script']) > max_script_chars:
            result['script'] = result['script'][:max_script_chars] + '...'
        return result

    except Exception as e:
        logger.error(f"Podcast script generation failed: {e}")
        return {
            "script": f"Script generation failed: {str(e)}",
            "segments": [],
            "estimated_duration": "0:00"
        }


def format_script_for_tts(segments: List[Dict]) -> str:
    """
    Format script segments for TTS processing.
    
    Args:
        segments: List of script segments with speaker and text
        
    Returns:
        Formatted script text for audio generation
    """
    formatted_text = ""
    
    for segment in segments:
        speaker = segment.get('speaker', 'Speaker')
        text = segment.get('text', '')
        
        # Add speaker identification and natural pauses
        formatted_text += f"{text}\n\n"
    
    return formatted_text.strip()


@router.post("/generate_script", response_model=PodcastScriptResponse)
async def generate_script(request: PodcastScriptRequest):
    """
    Generate a podcast script from session documents.
    
    Args:
        request: Script generation parameters
        
    Returns:
        Generated podcast script with segments
    """
    # Validate session
    session = storage_service.get_session(request.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {request.session_id}"
        )
    
    if not request.topic.strip():
        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty"
        )
    
    try:
        # Generate script using AI
        script_data = await generate_podcast_script(
            request.session_id,
            request.topic,
            request.style,
            request.max_sections
        )
        
        return PodcastScriptResponse(
            topic=request.topic,
            style=request.style,
            script=script_data.get('script', ''),
            segments=script_data.get('segments', []),
            estimated_duration=script_data.get('estimated_duration', '0:00'),
            message="Script generated successfully"
        )
        
    except Exception as e:
        logger.error(f"Script generation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Script generation failed: {str(e)}"
        )


@router.post("/generate_audio", response_model=PodcastGenerationResponse)
async def generate_audio(request: PodcastGenerationRequest, background_tasks: BackgroundTasks):
    """
    Generate podcast audio from script or topic.
    
    Args:
        request: Podcast generation parameters
        background_tasks: FastAPI background tasks
        
    Returns:
        Audio generation response with file path
    """
    # Validate session
    session = storage_service.get_session(request.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {request.session_id}"
        )
    
    try:
        script_text = ""
        current_pdf = getattr(request, 'current_pdf', None)  # Expect frontend to send current_pdf if possible

        # Use provided script or generate one
        if request.script:
            script_text = request.script
        elif request.topic:
            # Generate script first
            script_data = await generate_podcast_script(
                request.session_id,
                request.topic,
                "conversational",
                5,
                current_pdf=current_pdf
            )
            script_text = script_data.get('script', '')
            # If script is a fallback or error message, do not generate audio
            fallback_msgs = [
                "No relevant content found for the specified topic.",
                "AI script generation not available. Please configure GEMINI_API_KEY.",
            ]
            if not script_text or any(msg in script_text for msg in fallback_msgs) or script_text.startswith("Script generation failed"):
                raise HTTPException(
                    status_code=400,
                    detail="Could not generate a valid podcast script for this topic. Please try a different topic or check your documents."
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'script' or 'topic' must be provided"
            )

        if not script_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Script text cannot be empty"
            )

        # Truncate script to 1500 chars for safety
        script_text = script_text[:1500]

        # Force Azure as the TTS provider
        audio_result = await tts_service.generate_speech(
            text=script_text,
            voice=request.voice or "en-US-JennyNeural",
            provider="azure"
        )
        
        if not audio_result.get('success'):
            raise HTTPException(
                status_code=500,
                detail=f"Audio generation failed: {audio_result.get('error', 'Unknown error')}"
            )
        
        logger.info(f"Generated podcast audio for session {request.session_id}: {audio_result.get('file_path')}")
        logger.info(f"Audio result details: {audio_result}")
        
        return PodcastGenerationResponse(
            session_id=request.session_id,
            topic=request.topic or 'Custom Script',
            audio_file_path=audio_result.get('file_path'),
            script_used=script_text[:200] + "..." if len(script_text) > 200 else script_text,
            voice_used=request.voice or "en-US-JennyNeural",
            provider_used="azure",
            estimated_duration=audio_result.get('duration', '0:00'),
            message="Podcast audio generated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Podcast generation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Podcast generation failed: {str(e)}"
        )


@router.get("/audio/{session_id}/{filename}")
async def get_podcast_audio(session_id: str, filename: str, range: Optional[str] = Header(None)):
    """
    Retrieve generated podcast audio file.
    
    Args:
        session_id: Session ID
        filename: Audio filename
        
    Returns:
        Audio file response
    """
    # Validate session
    session = storage_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )
    
    # Look for the audio file in the temp directory
    # TODO: Implement proper session-scoped audio file storage
    import glob
    temp_dir = settings.TEMP_DIR
    
    logger.info(f"Looking for audio file: {filename} in directory: {temp_dir}")
    
    # First try exact filename match
    exact_path = os.path.join(temp_dir, filename)
    if os.path.exists(exact_path):
        audio_file_path = exact_path
        logger.info(f"Found exact audio file: {audio_file_path}")
    else:
        # Search for all podcast files and find the most recent one
        audio_files = glob.glob(os.path.join(temp_dir, f"podcast_*.wav"))
        logger.info(f"Available audio files: {audio_files}")
        
        if audio_files:
            # Return the most recent file regardless of exact filename match
            audio_files.sort(key=os.path.getmtime, reverse=True)
            audio_file_path = audio_files[0]  # Most recent
            logger.info(f"Using most recent audio file: {audio_file_path} for request: {filename}")
        else:
            logger.warning(f"No audio files found in {temp_dir}")
            audio_file_path = None
    
    if not audio_file_path or not os.path.exists(audio_file_path):
        raise HTTPException(
            status_code=404,
            detail="Audio file not found"
        )
    
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
        'Access-Control-Allow-Headers': 'Range, Content-Range',
        'Access-Control-Expose-Headers': 'Content-Range, Accept-Ranges, Content-Length'
    }
    
    # Determine media type based on file extension
    media_type = "audio/wav"
    if filename.lower().endswith('.mp3'):
        media_type = "audio/mpeg"
    elif filename.lower().endswith('.ogg'):
        media_type = "audio/ogg"
    
    return FileResponse(
        path=audio_file_path,
        filename=filename,
        media_type=media_type,
        headers=headers
    )


@router.options("/audio/{session_id}/{filename}")
async def options_podcast_audio(session_id: str, filename: str):
    """OPTIONS request for CORS preflight on audio files."""
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
        'Access-Control-Allow-Headers': 'Range, Content-Range, Content-Type',
        'Access-Control-Expose-Headers': 'Content-Range, Accept-Ranges, Content-Length',
        'Access-Control-Max-Age': '86400'  # 24 hours
    }
    return {"message": "CORS preflight"}, 200, headers


@router.get("/voices")
async def get_available_voices():
    """
    Get available TTS voices and providers.
    
    Returns:
        Available voices grouped by provider
    """
    return await tts_service.get_available_voices()


@router.get("/history/{session_id}")
async def get_podcast_history(session_id: str):
    """
    Get podcast generation history for a session.
    
    Args:
        session_id: Session to get history for
        
    Returns:
        List of generated podcasts
    """
    # Validate session
    session = storage_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )
    
    # For now, return empty history since we're not storing in session metadata yet
    # TODO: Implement proper podcast history storage
    podcasts = []
    
    return {
        "session_id": session_id,
        "podcasts": podcasts,
        "total_count": len(podcasts)
    }


@router.delete("/audio/{session_id}/{filename}")
async def delete_podcast_audio(session_id: str, filename: str):
    """
    Delete a generated podcast audio file.
    
    Args:
        session_id: Session ID
        filename: Audio filename to delete
        
    Returns:
        Deletion confirmation
    """
    # Validate session
    session = storage_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )
    
    try:
        # For now, just delete the file directly from temp directory
        # TODO: Implement proper session-scoped audio management
        import glob
        temp_dir = settings.TEMP_DIR
        
        # Find and delete the audio file
        audio_files = glob.glob(os.path.join(temp_dir, f"podcast_*.wav"))
        deleted = False
        
        for audio_file in audio_files:
            if filename in os.path.basename(audio_file):
                os.remove(audio_file)
                deleted = True
                break
        
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail="Audio file not found"
            )
        
        return {
            "message": "Podcast audio deleted successfully",
            "deleted_file": filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Podcast deletion error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete podcast: {str(e)}"
        )
