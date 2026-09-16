"""
NeMo Guardrails wrapper: input rail before retrieval, output rails after generation.

Why guardrails on a RAG app?
  - Input rail  : reject prompt-injection / jailbreak / harmful / secret-fishing
                  requests BEFORE they hit retrieval and the generator.
  - Output rails: (a) block unsafe or policy-violating answers, and
                  (b) "self check facts" — a hallucination guard that asks the
                  LLM whether the answer is actually supported by the
                  retrieved chunks. If not, the user gets a safe fallback
                  instead of a confident-sounding fabrication.

NeMo Guardrails normally drives the whole conversation itself; here we use
it as a library and run ONLY the rails (dialog/retrieval rails disabled),
keeping LangGraph in charge of the pipeline. The judge LLM is Gemini
(gemini_client.JUDGE_MODEL, a fast Flash-Lite model by default), wrapped in
LangChain so NeMo can call it.

Set GUARDRAILS=0 to disable (graph.build_graph(guardrails=False)).
"""

import os
from dataclasses import dataclass, field

from langchain_google_genai import ChatGoogleGenerativeAI
from nemoguardrails import LLMRails, RailsConfig

import gemini_client

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guardrails")

# Run only input rails / only output rails; never let NeMo generate the answer.
_INPUT_ONLY = {"rails": {"output": False, "dialog": False, "retrieval": False},
               "log": {"activated_rails": True}}
_OUTPUT_ONLY = {"rails": {"input": False, "dialog": False, "retrieval": False},
                "log": {"activated_rails": True}}


@dataclass
class GuardResult:
    allowed: bool
    blocked_by: str | None = None      # e.g. "self check input", "self check facts"
    message: str | None = None         # refusal text from the rails, when blocked
    activated: list[str] = field(default_factory=list)  # every rail that ran


class _GeminiJudge(ChatGoogleGenerativeAI):
    """
    NeMo's LangChain adapter binds OpenAI-style `max_tokens=` onto the model
    for every rail call; Gemini's GenerateContentConfig only knows
    `max_output_tokens` and rejects unknown keys. Translate it.
    """

    @staticmethod
    def _fix(kwargs: dict) -> dict:
        if "max_tokens" in kwargs:
            kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
        return kwargs

    def _generate(self, *args, **kwargs):
        return super()._generate(*args, **self._fix(kwargs))

    async def _agenerate(self, *args, **kwargs):
        return await super()._agenerate(*args, **self._fix(kwargs))


class Guard:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        judge_llm = _GeminiJudge(
            model=gemini_client.JUDGE_MODEL, google_api_key=api_key, temperature=0.0
        )
        config = RailsConfig.from_path(CONFIG_DIR)
        self.rails = LLMRails(config, llm=judge_llm)

    @staticmethod
    def _summarize(result) -> GuardResult:
        activated = [r.name for r in (result.log.activated_rails or [])]
        blocker = next((r for r in (result.log.activated_rails or []) if r.stop), None)
        if blocker:
            text = result.response[-1]["content"] if result.response else None
            return GuardResult(False, blocker.name, text, activated)
        return GuardResult(True, None, None, activated)

    def check_input(self, question: str) -> GuardResult:
        """Input rail: is this question allowed to proceed to retrieval?"""
        res = self.rails.generate(
            messages=[{"role": "user", "content": question}], options=_INPUT_ONLY
        )
        return self._summarize(res)

    def check_output(self, question: str, answer: str, chunks: list[str]) -> GuardResult:
        """
        Output rails: is the generated answer safe, and is it supported by the
        retrieved chunks? `chunks` become $relevant_chunks for `self check facts`.
        """
        res = self.rails.generate(
            messages=[
                {"role": "context", "content": {
                    "relevant_chunks": "\n\n".join(chunks),
                    "check_facts": True,
                }},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            options=_OUTPUT_ONLY,
        )
        return self._summarize(res)
