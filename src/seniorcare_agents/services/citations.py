from seniorcare_agents.models import Citation, RetrievedChunk


class CitationService:
    def from_chunks(self, chunks: list[RetrievedChunk]) -> list[Citation]:
        citations: list[Citation] = []
        seen = set()
        for chunk in chunks:
            key = (chunk.source_url, chunk.title, chunk.page_number)
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                Citation(
                    citation_id=f"SRC{len(citations) + 1}",
                    source_name=chunk.source_name,
                    title=chunk.title,
                    program_name=chunk.program_name,
                    source_url=chunk.source_url,
                    page_number=chunk.page_number,
                    last_verified=chunk.last_verified,
                )
            )
        return citations
