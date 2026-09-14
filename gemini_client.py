"""Thin wrapper around the google-genai SDK for embeddings + generation."""

import json
import mimetypes
import os
from google import genai
from google.genai import types

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; skip if not installed

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
CHAT_MODEL = "gemini-flash-latest"  # supports native audio, image + video input

def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set the GEMINI_API_KEY environment variable. "
            "Get a key at https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=api_key)


def embed_text2(client: genai.Client, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """task_type differs for docs vs queries — this improves retrieval quality."""
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return result.embeddings[0].values

def embed_text(client: genai.Client, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """task_type differs for docs vs queries — this improves retrieval quality.
    output_dimensionality is set explicitly so vectors match Milvus's schema
    (EMBED_DIM=768) — gemini-embedding-001 defaults to 3072-dim otherwise."""
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBED_DIM,
        ),
    )
    return result.embeddings[0].values

def _audio_mime(path: str) -> str:
    """Browser mic recordings arrive as .webm, which mimetypes reports as
    video/webm — Gemini wants audio/webm for those."""
    if path.lower().endswith(".webm"):
        return "audio/webm"
    return mimetypes.guess_type(path)[0] or "audio/wav"


def transcribe_audio(client: genai.Client, audio_path: str) -> str:
    """
    Turns an audio file into text using Gemini's native audio understanding.
    This is the "multimodal" step: Gemini can listen to the clip directly
    (no separate speech-to-text service needed). The returned transcript is
    then embedded and BM25-indexed exactly like a text document — so the
    rest of the pipeline (Milvus, retrieval, generation) never has to know
    the source was audio.

    Works for files up to ~20MB using inline bytes. For larger files, use
    client.files.upload() instead and pass the returned file reference.
    """
    mime_type = mimetypes.guess_type(audio_path)[0] or "audio/wav"
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    prompt = (
        "Transcribe this audio clip accurately. Return only the transcript "
        "text, with no preamble or extra commentary."
    )
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            prompt,
        ],
    )
    return response.text.strip()


def transcribe_query(client: genai.Client, audio_path: str) -> str:
    """
    Transcribes a *spoken question* (as opposed to a document clip).

    Same Gemini native-audio call as transcribe_audio(), but the prompt asks
    for a clean, search-ready question: spelled-out identifiers such as
    "E R R dash four five two one" are normalized to "ERR-4521" so the BM25
    (keyword) leg of the hybrid search can still match exact codes.
    """
    mime_type = _audio_mime(audio_path)
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    prompt = (
        "This audio clip is a user asking a question to a search system. "
        "Transcribe the question as a single clean sentence. Normalize any "
        "spelled-out codes, identifiers or numbers into their written form "
        "(e.g. 'E R R dash four five two one' -> 'ERR-4521'). Return only "
        "the question text, no preamble."
    )
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            prompt,
        ],
    )
    return response.text.strip()


def _image_part(image_path: str) -> types.Part:
    mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as f:
        return types.Part.from_bytes(data=f.read(), mime_type=mime_type)


def describe_image(client: genai.Client, image_path: str) -> str:
    """
    Turns an image into searchable text using Gemini's native vision.

    Same idea as transcribe_audio(): the *description* is what gets embedded
    and BM25-indexed, so the prompt asks for every piece of visible text
    verbatim (codes, SKUs, numbers) plus a short description of what the
    image shows. That way exact-keyword search still works on screenshots,
    labels and diagrams, and semantic search works on photos.
    """
    prompt = (
        "Describe this image for a search index. First, transcribe ALL visible "
        "text exactly as written (error codes, SKUs, numbers, labels). Then, in "
        "one or two sentences, describe what the image shows. Be factual and "
        "concise; plain text only (no markdown, no headings), no preamble. "
        "Keep it under 200 words."
    )
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=[_image_part(image_path), prompt],
    )
    return response.text.strip()


def describe_image_query(client: genai.Client, image_path: str, text: str | None = None) -> str:
    """
    Turns an image (plus optional typed text) into a search-ready question.

    Counterpart of transcribe_query(): if the user typed a question alongside
    the image ("what does this mean?"), Gemini grounds that question in the
    image's content; if not, it infers the most likely question.
    """
    if text:
        prompt = (
            f"The user attached this image and asked: \"{text}\"\n"
            "Rewrite their question as a single self-contained sentence that "
            "includes the key details from the image (any visible codes, "
            "identifiers, names or numbers written exactly as shown). Return "
            "only the rewritten question."
        )
    else:
        prompt = (
            "The user attached this image as a question to a search system, "
            "without typing anything. Write the single most likely question "
            "they are asking, including the key details visible in the image "
            "(codes, identifiers, names, numbers written exactly as shown). "
            "Return only the question."
        )
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=[_image_part(image_path), prompt],
    )
    return response.text.strip()


INLINE_LIMIT_BYTES = 20 * 1024 * 1024  # Gemini inline-bytes limit; larger files go via the Files API


def _video_part(client: genai.Client, video_path: str):
    """
    Small clips are sent inline; bigger ones are uploaded through the Files
    API (which returns a reference Gemini can read directly).
    """
    mime_type = mimetypes.guess_type(video_path)[0] or "video/mp4"
    if os.path.getsize(video_path) <= INLINE_LIMIT_BYTES:
        with open(video_path, "rb") as f:
            return types.Part.from_bytes(data=f.read(), mime_type=mime_type)
    uploaded = client.files.upload(file=video_path, config={"mime_type": mime_type})
    # Files API processes video asynchronously; poll until it's ready.
    import time
    while uploaded.state and uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    return uploaded


def describe_video(client: genai.Client, video_path: str) -> str:
    """
    Turns a video into searchable text using Gemini's native video
    understanding (it sees the frames AND hears the audio track).

    Same idea as transcribe_audio()/describe_image(): the returned text is
    what gets embedded and BM25-indexed. The prompt asks for the spoken
    narration verbatim, every piece of on-screen text verbatim, and a short
    summary of what happens — so exact codes/numbers from either the audio
    or the visuals stay keyword-searchable.
    """
    prompt = (
        "Describe this video for a search index. Include: (1) a transcript of "
        "everything spoken, (2) all on-screen text exactly as written (codes, "
        "numbers, names, dates), and (3) a two-sentence summary of what the "
        "video shows. Plain text only, no markdown or headings, no preamble. "
        "Keep it under 300 words."
    )
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=[_video_part(client, video_path), prompt],
    )
    return response.text.strip()


def describe_video_query(client: genai.Client, video_path: str, text: str | None = None) -> str:
    """
    Turns a video (plus optional typed text) into a search-ready question —
    e.g. a screen recording of a bug with the user narrating "what is this?".
    Counterpart of transcribe_query()/describe_image_query().
    """
    if text:
        prompt = (
            f"The user attached this video and asked: \"{text}\"\n"
            "Rewrite their question as a single self-contained sentence that "
            "includes the key details from the video (anything spoken, and any "
            "visible codes, identifiers, names or numbers written exactly as "
            "shown). Return only the rewritten question."
        )
    else:
        prompt = (
            "The user attached this video as a question to a search system. "
            "If they speak a question in it, use that; otherwise infer the most "
            "likely question. Write it as a single self-contained sentence that "
            "includes the key details visible or spoken in the video (codes, "
            "identifiers, names, numbers written exactly as shown). Return only "
            "the question."
        )
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=[_video_part(client, video_path), prompt],
    )
    return response.text.strip()


def rerank(client: genai.Client, question: str, candidates: list[str], top_n: int = 5) -> list[dict]:
    """
    LLM re-ranker: scores every candidate chunk for relevance to the question
    in ONE listwise Gemini call, returning [{"index": i, "score": 0-10}, ...]
    sorted best-first and truncated to top_n.

    Why a second ranking stage? Hybrid search (dense + BM25 fused with RRF)
    is cheap and recall-oriented: it decides *which* ~10 chunks are worth
    looking at from thousands. A cross-attention model that reads the
    question and each chunk *together* is far better at deciding which of
    those 10 actually answer the question — but it is too expensive to run
    over the whole collection. So: retrieve wide, re-rank narrow.

    Because every modality has already been turned into text upstream, the
    same re-ranker works for text, audio, image and video chunks.
    Structured output (response_schema) guarantees parseable JSON.
    """
    if not candidates:
        return []
    numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(candidates))
    prompt = (
        "You are a search re-ranker. Score how well EACH passage answers the "
        "question, on a 0-10 scale (10 = directly and completely answers it, "
        "0 = unrelated). Judge every passage independently; passages that "
        "merely share keywords with the question but do not answer it should "
        "score low.\n\n"
        f"Question: {question}\n\nPassages:\n{numbered}"
    )
    schema = types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "index": types.Schema(type=types.Type.INTEGER),
                "score": types.Schema(type=types.Type.NUMBER),
            },
            required=["index", "score"],
        ),
    )
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.0,
        ),
    )
    scored = {int(r["index"]): float(r["score"]) for r in json.loads(response.text)}
    # Any index the model skipped gets 0 so nothing silently disappears.
    ranked = [{"index": i, "score": scored.get(i, 0.0)} for i in range(len(candidates))]
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked[:top_n]


def generate_answer(client: genai.Client, question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
    prompt = (
        "Answer the question using ONLY the information in the context below. "
        "The context may phrase things differently from the question — use any "
        "passage that addresses the same code, product, event or topic, and "
        "cite sources like [1], [2]. Only say the context lacks the answer if "
        "no passage is relevant at all.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    response = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    return response.text