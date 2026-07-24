"""Deterministic offline stand-in for answer generation.

NOT a language model -- it does simple extractive selection: given the
retrieved context chunks, it picks the chunk with the highest word overlap
with the question and returns it verbatim with a citation tag, or says it
doesn't know if no chunk overlaps at all. This is enough to exercise the
prompt-building, citation-formatting, and "don't answer beyond the context"
guardrail in rag.py and its tests without any API key. Swap LLM_PROVIDER to
openai/anthropic for real grounded generation.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_CHUNK_RE = re.compile(r"\[(?P<cid>[^\]]+)\]\s*(?P<text>.*?)(?=\n\[|\Z)", re.DOTALL)


class MockChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:  # pragma: no cover - trivial
        return "mock-rag-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = messages[-1].content
        reply = self._answer(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])

    def _answer(self, prompt: str) -> str:
        question = _after_marker(prompt, "QUESTION:", until="CONTEXT:").strip()
        context_block = _after_marker(prompt, "CONTEXT:").strip()

        chunks = _CHUNK_RE.findall(context_block)
        if not chunks:
            return "I don't have enough information in the provided context to answer that."

        q_words = set(_words(question))
        best_cid, best_text, best_overlap = None, None, 0
        for cid, text in chunks:
            overlap = len(q_words & set(_words(text)))
            if overlap > best_overlap:
                best_cid, best_text, best_overlap = cid, text.strip(), overlap

        if best_overlap == 0 or best_cid is None:
            return "I don't have enough information in the provided context to answer that."

        return f"{best_text} [{best_cid}]"


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "and", "or", "but", "with",
    "what", "which", "who", "whom", "how", "why", "when", "where",
    "does", "do", "did", "this", "that", "these", "those", "it", "its",
}


def _words(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


def _after_marker(prompt: str, marker: str, until: Optional[str] = None) -> str:
    idx = prompt.find(marker)
    if idx == -1:
        return ""
    rest = prompt[idx + len(marker):]
    stops = [s for s in (rest.find("###"), rest.find(until) if until else -1) if s != -1]
    end = min(stops) if stops else -1
    return rest if end == -1 else rest[:end]
