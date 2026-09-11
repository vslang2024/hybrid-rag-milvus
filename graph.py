"""LangGraph pipeline: [transcribe_question | interpret_image | interpret_video] -> retrieve -> [rerank] -> generate.

The query can arrive as text (`question`), an audio file (`question_audio`),
an image (`question_image`) or a video (`question_video`) — the latter two
optionally with `question` as a caption. Non-text queries are turned into a
text question by Gemini first, then follow the exact same retrieve ->
rerank -> generate path as typed questions.

Retrieval is two-stage: hybrid search (dense + BM25, RRF-fused) pulls a wide
candidate pool, then a Gemini LLM re-ranker reads question + each candidate
together and keeps the best few for the generator. build_graph(rerank=False)
skips the second stage so you can compare.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END

import gemini_client
import milvus_store


class RAGState(TypedDict, total=False):
    question: str            # text query (filled in from audio if question_audio is given)
    question_audio: str      # optional path to a spoken question (.wav/.mp3/.m4a)
    question_image: str      # optional path to an image query (.png/.jpg); `question` = caption
    question_video: str      # optional path to a video query (.mp4/.mov/.webm); `question` = caption
    candidates: list[dict]   # hybrid-search results (wide pool), before re-ranking
    retrieved: list[dict]    # what the generator sees: re-ranked top-N (or candidates[:top_k] if rerank=False)
    answer: str


CANDIDATE_POOL = 10   # how many hybrid hits to hand to the re-ranker
FINAL_TOP_K = 5       # how many chunks the generator sees


def build_graph(milvus_client, genai_client, rerank: bool = True):
    def transcribe_question_node(state: RAGState) -> RAGState:
        """Audio query -> text query, via Gemini native audio understanding."""
        question = gemini_client.transcribe_query(genai_client, state["question_audio"])
        return {"question": question}

    def interpret_image_node(state: RAGState) -> RAGState:
        """Image (+ optional caption) -> text query, via Gemini native vision."""
        question = gemini_client.describe_image_query(
            genai_client, state["question_image"], text=state.get("question")
        )
        return {"question": question}

    def interpret_video_node(state: RAGState) -> RAGState:
        """Video (+ optional caption) -> text query, via Gemini native video understanding."""
        question = gemini_client.describe_video_query(
            genai_client, state["question_video"], text=state.get("question")
        )
        return {"question": question}

    def retrieve_node(state: RAGState) -> RAGState:
        question = state["question"]
        query_vector = gemini_client.embed_text(
            genai_client, question, task_type="RETRIEVAL_QUERY"
        )
        pool = CANDIDATE_POOL if rerank else FINAL_TOP_K
        hits = milvus_store.hybrid_search(
            milvus_client, query_text=question, query_dense_vector=query_vector, top_k=pool
        )
        for rank, h in enumerate(hits, start=1):
            h["hybrid_rank"] = rank
        return {"candidates": hits, "retrieved": hits[:FINAL_TOP_K]}

    def rerank_node(state: RAGState) -> RAGState:
        """Second-stage ranking: Gemini scores each candidate against the question."""
        candidates = state["candidates"]
        ranked = gemini_client.rerank(
            genai_client, state["question"], [c["text"] for c in candidates], top_n=FINAL_TOP_K
        )
        retrieved = []
        for r in ranked:
            hit = dict(candidates[r["index"]])
            hit["rerank_score"] = r["score"]
            retrieved.append(hit)
        # Drop chunks the re-ranker judged irrelevant (score 0) so the
        # generator isn't handed keyword-only noise — unless that's all we have.
        kept = [h for h in retrieved if h["rerank_score"] > 0]
        return {"retrieved": kept or retrieved}

    def generate_node(state: RAGState) -> RAGState:
        chunks = [h["text"] for h in state["retrieved"]]
        answer = gemini_client.generate_answer(genai_client, state["question"], chunks)
        return {"answer": answer}

    def route_input(state: RAGState) -> str:
        """Pick the entry node based on whether the query is audio or text."""
        if state.get("question_video"):
            return "interpret_video"
        if state.get("question_image"):
            return "interpret_image"
        if state.get("question_audio"):
            return "transcribe_question"
        if state.get("question"):
            return "retrieve"
        raise ValueError(
            "Provide 'question' (text), 'question_audio', 'question_image' or 'question_video' (file path)."
        )

    graph = StateGraph(RAGState)
    graph.add_node("transcribe_question", transcribe_question_node)
    graph.add_node("interpret_image", interpret_image_node)
    graph.add_node("interpret_video", interpret_video_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)

    graph.set_conditional_entry_point(
        route_input,
        {
            "transcribe_question": "transcribe_question",
            "interpret_image": "interpret_image",
            "interpret_video": "interpret_video",
            "retrieve": "retrieve",
        },
    )
    graph.add_edge("transcribe_question", "retrieve")
    graph.add_edge("interpret_image", "retrieve")
    graph.add_edge("interpret_video", "retrieve")
    if rerank:
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "generate")
    else:
        graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
