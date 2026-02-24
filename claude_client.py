"""Claude Citations API client for answer generation with native citations.

Supports two modes:
- Native Citations API (requires Claude Sonnet 3.5+ or Claude 4+)
- Prompt-based citations fallback (works with any Claude model)

The mode is auto-detected based on model capabilities.
"""

import json
import re

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

# Models that support native Citations API
CITATION_MODELS = {
    "claude-sonnet-4-5-20250514",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-20241022",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-haiku-4-5-20251001",
}


def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _model_supports_citations(model: str) -> bool:
    return model in CITATION_MODELS


def build_document_blocks(chunks: list[dict], enable_citations: bool = True) -> list[dict]:
    """Convert retrieved chunks into Claude document content blocks."""
    blocks = []
    for idx, chunk in enumerate(chunks):
        title = f"{chunk.get('company', 'Unknown')} - {chunk.get('section', 'unknown')} (Chunk {chunk.get('chunk_index', idx)})"
        block = {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": chunk["text"],
            },
            "title": title,
            "context": (
                f"SEC 10-K Filing | Company: {chunk.get('company', 'N/A')} "
                f"| Section: {chunk.get('section', 'N/A')} "
                f"| Fiscal Year: {chunk.get('fiscal_year', 'N/A')}"
            ),
        }
        if enable_citations:
            block["citations"] = {"enabled": True}
        blocks.append(block)
    return blocks


def _generate_with_native_citations(client: anthropic.Anthropic, query: str, chunks: list[dict], model: str) -> dict:
    """Generate answer using Claude's native Citations API."""
    document_blocks = build_document_blocks(chunks, enable_citations=True)

    system_prompt = (
        "You are a senior financial analyst at a top investment bank. "
        "You have been given excerpts from SEC 10-K filings as reference documents. "
        "Your job: answer the user's question using these documents. "
        "RULES: "
        "1. NEVER refuse to answer. NEVER say 'I cannot answer' or 'I would need more documents'. "
        "2. ALWAYS provide a substantive, analytical answer based on whatever evidence is available. "
        "3. For comparison questions: directly compare the companies using the specific risk factors, strategies, and data points found in the documents. State which company appears stronger or weaker on each dimension and why. "
        "4. Cite specific passages to support every claim. "
        "5. If documents are incomplete, analyze what IS there. Do not complain about what is missing. "
        "6. Be opinionated. Take a clear analytical stance supported by the evidence."
    )

    # Add cache_control to the last document block for prompt caching
    # This caches the system prompt + all document blocks across requests
    if document_blocks:
        document_blocks[-1]["cache_control"] = {"type": "ephemeral"}

    user_text = (
        f"{query}\n\n"
        "Analyze the documents above and provide a direct, substantive answer. "
        "Do not hedge or say you cannot answer. Cite specific passages."
    )
    content = document_blocks + [{"type": "text", "text": user_text}]

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": content}],
    )

    answer_text = ""
    citations = []
    citation_idx = 0

    for block in response.content:
        if block.type == "text":
            if hasattr(block, "citations") and block.citations:
                for cite in block.citations:
                    citation_idx += 1
                    entry = {
                        "index": citation_idx,
                        "cited_text": getattr(cite, "cited_text", ""),
                        "document_index": getattr(cite, "document_index", 0),
                        "document_title": getattr(cite, "document_title", ""),
                        "start_char_index": getattr(cite, "start_char_index", 0),
                        "end_char_index": getattr(cite, "end_char_index", 0),
                    }
                    doc_idx = entry["document_index"]
                    if doc_idx < len(chunks):
                        source = chunks[doc_idx]
                        entry["source_company"] = source.get("company", "")
                        entry["source_section"] = source.get("section", "")
                        entry["source_fiscal_year"] = source.get("fiscal_year", "")
                        entry["source_ticker"] = source.get("ticker", "")
                    citations.append(entry)
                # Insert citation markers into text
                answer_text += block.text
            else:
                answer_text += block.text

    usage = response.usage
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0)
    cache_read = getattr(usage, "cache_read_input_tokens", 0)
    total_input = usage.input_tokens + cache_creation + cache_read

    return {
        "answer": answer_text,
        "citations": citations,
        "citation_mode": "native",
        "model": model,
        "input_tokens": total_input,
        "output_tokens": usage.output_tokens,
        "cache_creation_tokens": cache_creation,
        "cache_read_tokens": cache_read,
    }


def _generate_with_prompt_citations(client: anthropic.Anthropic, query: str, chunks: list[dict], model: str) -> dict:
    """Generate answer with prompt-based citations (fallback for older models)."""
    # Build context with numbered documents
    context_parts = []
    for idx, chunk in enumerate(chunks):
        header = (
            f"[Document {idx + 1}] "
            f"{chunk.get('company', 'Unknown')} | "
            f"{chunk.get('section', 'unknown')} | "
            f"FY{chunk.get('fiscal_year', 'N/A')}"
        )
        context_parts.append(f"{header}\n{chunk['text']}")

    documents_text = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a financial analyst assistant specializing in SEC 10-K filings. "
        "Answer questions using ONLY the provided documents.\n\n"
        "IMPORTANT: You must cite your sources. For every claim, include a citation "
        "in the format [N] where N is the document number. After your answer, provide "
        "a CITATIONS section listing each citation with the exact quoted text.\n\n"
        "Format your response as:\n"
        "ANSWER:\n<your answer with [N] citations inline>\n\n"
        "CITATIONS:\n"
        "[1] \"exact quoted text from document\" (Source: Company, Section)\n"
        "[2] \"exact quoted text\" (Source: Company, Section)\n"
        "...\n\n"
        "Be precise, factual, and reference specific data points when available."
    )

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": f"Documents:\n\n{documents_text}\n\nQuestion: {query}",
        }],
    )

    raw_text = response.content[0].text

    # Parse the structured response
    answer_text = raw_text
    citations = []

    # Try to split into ANSWER and CITATIONS sections
    if "CITATIONS:" in raw_text:
        parts = raw_text.split("CITATIONS:", 1)
        answer_text = parts[0].replace("ANSWER:", "").strip()
        citations_text = parts[1].strip()

        # Parse citation lines: [N] "quoted text" (Source: Company, Section)
        cite_pattern = r'\[(\d+)\]\s*["\u201c](.+?)["\u201d]\s*\((?:Source:\s*)?(.+?)\)'
        for match in re.finditer(cite_pattern, citations_text, re.DOTALL):
            cite_num = int(match.group(1))
            cited_text = match.group(2).strip()
            source_info = match.group(3).strip()

            doc_idx = cite_num - 1  # Convert 1-indexed to 0-indexed
            entry = {
                "index": cite_num,
                "cited_text": cited_text,
                "document_index": doc_idx,
                "document_title": "",
                "source_info": source_info,
            }

            if 0 <= doc_idx < len(chunks):
                source = chunks[doc_idx]
                entry["source_company"] = source.get("company", "")
                entry["source_section"] = source.get("section", "")
                entry["source_fiscal_year"] = source.get("fiscal_year", "")
                entry["source_ticker"] = source.get("ticker", "")
                entry["document_title"] = (
                    f"{source.get('company', 'Unknown')} - "
                    f"{source.get('section', 'unknown')}"
                )

            citations.append(entry)

    return {
        "answer": answer_text,
        "citations": citations,
        "citation_mode": "prompt",
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def generate_answer(query: str, chunks: list[dict], model: str | None = None) -> dict:
    """Generate an answer using Claude with citations.

    Automatically selects native Citations API or prompt-based fallback
    depending on the model's capabilities.

    Returns dict with: answer, citations, citation_mode, input_tokens, output_tokens
    """
    client = get_client()
    model = model or CLAUDE_MODEL

    if _model_supports_citations(model):
        try:
            return _generate_with_native_citations(client, query, chunks, model)
        except anthropic.BadRequestError:
            # Model might not actually support citations despite being in the list
            pass

    return _generate_with_prompt_citations(client, query, chunks, model)
