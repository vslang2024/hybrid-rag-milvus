"""
Chat server for the hybrid RAG pipeline.

Exposes the LangGraph app over HTTP so a user can ask questions from a
browser — either by typing, uploading an audio file, or recording with the
microphone. Both paths go through the same graph as main.py:

    text  -> /api/chat        -> {"question": ...}
    audio -> /api/chat/audio  -> {"question_audio": <temp file>}
    image -> /api/chat/image  -> {"question_image": <temp file>, "question": <caption>}
    video -> /api/chat/video  -> {"question_video": <temp file>, "question": <caption>}

The server reuses the collection built by main.py — in local hybrid_rag.db
(Milvus Lite) by default, or in Milvus Standalone when MILVUS_ADDRESS is set.
Run main.py once first (or POST /api/ingest) to populate it.

Usage:
    uvicorn server:app --reload                       # Milvus Lite
    MILVUS_ADDRESS=http://localhost:19530 uvicorn server:app   # Milvus Standalone
    open http://127.0.0.1:8000
"""

import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import gemini_client
import milvus_store
from graph import build_graph
from main import load_audio_docs, load_image_docs, load_text_docs, load_video_docs

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

state: dict = {}  # holds the clients + compiled graph for the process lifetime


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["genai"] = gemini_client.get_client()
    state["milvus"] = milvus_store.get_client()   # MILVUS_ADDRESS env var, or local hybrid_rag.db
    # A collection created by another process starts "released"; load it so
    # searches work (main.py never hits this because it creates + searches
    # in the same process).
    milvus_store.ensure_loaded(state["milvus"])
    use_rerank = os.environ.get("RERANK", "1") != "0"         # RERANK=0     -> no re-ranker
    use_guardrails = os.environ.get("GUARDRAILS", "1") != "0" # GUARDRAILS=0 -> no NeMo rails
    state["app"] = build_graph(
        state["milvus"], state["genai"], rerank=use_rerank, guardrails=use_guardrails
    )
    yield
    state.clear()


app = FastAPI(title="Hybrid RAG chat", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    question: str


def _ensure_indexed():
    if not state["milvus"].has_collection(milvus_store.COLLECTION_NAME):
        raise HTTPException(
            status_code=409,
            detail="No collection found. Run `python main.py` or POST /api/ingest first.",
        )


def _run(inputs: dict) -> dict:
    result = state["app"].invoke(inputs)
    return {
        "question": result["question"],
        "answer": result["answer"],
        "retrieved": result.get("retrieved", []),
        "candidates": result.get("candidates", []),
        "blocked_by": result.get("blocked_by"),
        "guardrails": result.get("guardrails", []),
    }


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Typed question."""
    _ensure_indexed()
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is empty")
    return _run({"question": q})


@app.post("/api/chat/audio")
async def chat_audio(file: UploadFile = File(...)):
    """Spoken question: uploaded .wav/.mp3/.m4a/.webm, transcribed by Gemini."""
    _ensure_indexed()
    suffix = os.path.splitext(file.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        return await run_in_threadpool(_run, {"question_audio": tmp_path})
    finally:
        os.unlink(tmp_path)


async def _run_with_file(file: UploadFile, key: str, default_suffix: str, caption: str = ""):
    """Save the upload to a temp file, invoke the graph with {key: path}, clean up."""
    suffix = os.path.splitext(file.filename or "")[1] or default_suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        inputs = {key: tmp_path}
        if caption.strip():
            inputs["question"] = caption.strip()
        # Sync graph (and NeMo's sync generate()) must not run on the event loop.
        return await run_in_threadpool(_run, inputs)
    finally:
        os.unlink(tmp_path)


@app.post("/api/chat/image")
async def chat_image(file: UploadFile = File(...), question: str = Form("")):
    """Image query (.png/.jpg/.webp) with an optional typed caption."""
    _ensure_indexed()
    return await _run_with_file(file, "question_image", ".png", question)


@app.post("/api/chat/video")
async def chat_video(file: UploadFile = File(...), question: str = Form("")):
    """Video query (.mp4/.mov/.webm) with an optional typed caption."""
    _ensure_indexed()
    return await _run_with_file(file, "question_video", ".mp4", question)


@app.post("/api/ingest")
def ingest():
    """(Re)build the collection from data/ — same steps as main.py."""
    milvus_store.create_collection(state["milvus"], drop_existing=True)
    docs = (
        load_text_docs()
        + load_audio_docs(state["genai"])
        + load_image_docs(state["genai"])
        + load_video_docs(state["genai"])
    )
    embed_fn = lambda t: gemini_client.embed_text(state["genai"], t, task_type="RETRIEVAL_DOCUMENT")
    milvus_store.insert_documents(state["milvus"], docs, embed_fn)
    return {"indexed": len(docs)}
