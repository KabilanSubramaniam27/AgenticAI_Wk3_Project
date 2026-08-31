from seniorcare_agents.models import RetrievedChunk


def reciprocal_rank_fusion(
    result_sets: list[list[RetrievedChunk]], rrf_k: int = 60
) -> list[RetrievedChunk]:
    merged: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}
    for rows in result_sets:
        for rank, row in enumerate(rows, 1):
            scores[row.chunk_id] = scores.get(row.chunk_id, 0.0) + 1 / (rrf_k + rank)
            if row.chunk_id not in merged:
                merged[row.chunk_id] = row.model_copy(deep=True)
            else:
                current = merged[row.chunk_id]
                current.bm25_score = current.bm25_score or row.bm25_score
                current.bm25_rank = current.bm25_rank or row.bm25_rank
                current.vector_score = current.vector_score or row.vector_score
                current.vector_rank = current.vector_rank or row.vector_rank
                current.retrieved_by = list(
                    dict.fromkeys([*current.retrieved_by, *row.retrieved_by])
                )
    ordered = sorted(merged.values(), key=lambda row: scores[row.chunk_id], reverse=True)
    for rank, row in enumerate(ordered, 1):
        row.fusion_score = scores[row.chunk_id]
        row.fusion_rank = rank
    return ordered
