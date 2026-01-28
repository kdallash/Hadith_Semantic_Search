import numpy as np


def query_anchor_scores(query, model, anchor_index, top_k=10):
    q_emb = model.encode(query, normalize_embeddings=True)
    scores, indices = anchor_index.search(q_emb.reshape(1, -1), top_k)
    return np.array(indices[0], dtype=int), np.array(scores[0], dtype=float)


def bm25_retrieve(query, bm25, preprocess_query, top_k=50):
    tokenized_query = preprocess_query(query)
    scores = bm25.get_scores(tokenized_query)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return top_idx, scores[top_idx]


def compute_anchor_scores_for_hadiths(
    n_hadiths,
    anchor_indices,
    anchor_scores,
    anchor_dict,
    unique_anchor_texts
):
    anchor_score_vec = np.zeros(n_hadiths, dtype=float)

    for a_idx, a_score in zip(anchor_indices, anchor_scores):
        if 0 <= a_idx < len(unique_anchor_texts):
            anchor_text = unique_anchor_texts[a_idx]
            for h_idx in anchor_dict.get(anchor_text, []):
                anchor_score_vec[h_idx] = a_score

    return anchor_score_vec



def hybrid_search_fixed(query,
                        df,
                        bm25,
                        preprocess_query,
                        model,
                        hadith_embeddings,
                        anchor_index,
                        anchor_dict,
                        unique_anchor_texts,
                        top_k=5,
                        top_bm25=50,
                        top_anchors=10,
                        alpha_anchor=0.40,
                        alpha_semantic=0.35,
                        alpha_bm25=0.25,
                        full_semantic=False):
    """
    Hybrid search with correct signal alignment:
    - bm25 retrieves top_bm25 hadiths + scores
    - anchor_index returns top_anchors anchors + scores -> mapped to hadith-level anchor scores
    - semantic scores computed either for full corpus (if full_semantic=True) OR for just the union of bm25 candidates + anchor-linked hadiths
    - missing-signal entries are zero
    """

    n = len(df)
    eps = 1e-8

    # 1) BM25 candidates and scores
    bm25_indices, bm25_scores = bm25_retrieve(query, bm25, preprocess_query,top_k=top_bm25)
    # make dictionary mapping hadith_idx -> bm25_score
    bm25_map = {int(idx): float(score) for idx, score in zip(bm25_indices, bm25_scores)}

    # 2) Anchor retrieval -> anchor indices + scores
    anchor_idx, anchor_scores = query_anchor_scores(query, model, anchor_index, top_k=top_anchors)

    # 3) Build per-hadith anchor score for entire corpus (zeros by default)
    anchor_score_vec = compute_anchor_scores_for_hadiths(
        n_hadiths=n,
        anchor_indices=anchor_idx,
        anchor_scores=anchor_scores,
        anchor_dict=anchor_dict,
        unique_anchor_texts=unique_anchor_texts,
    )

    # 4) Determine which hadith indices we will score semantically.
    # union of bm25 candidates and all anchor-linked hadiths returned
    anchor_linked_indices = []
    for a_idx in anchor_idx:
        # safe check
        if 0 <= a_idx < len(unique_anchor_texts):
            anchor_text = unique_anchor_texts[int(a_idx)]
            anchor_linked_indices.extend(anchor_dict.get(anchor_text, []))

    anchor_linked_indices = np.unique(np.array(anchor_linked_indices, dtype=int)) if len(anchor_linked_indices) else np.array([], dtype=int)

    if full_semantic:
        # compute semantic for whole corpus (slower)
        query_emb = model.encode(query, normalize_embeddings=True)
        # hadith_embeddings @ query_emb
        semantic_scores_all = hadith_embeddings @ query_emb
        semantic_score_vec = np.array(semantic_scores_all, dtype=float)
    else:
        # compute semantic only for union (faster)
        union_indices = np.unique(np.concatenate([bm25_indices, anchor_linked_indices])).astype(int)
        query_emb = model.encode(query, normalize_embeddings=True)
        # compute dot product for selected indices only
        if len(union_indices) > 0:
            sem_vals = hadith_embeddings[union_indices] @ query_emb
            # place into full-length vector
            semantic_score_vec = np.zeros(n, dtype=float)
            semantic_score_vec[union_indices] = sem_vals
        else:
            semantic_score_vec = np.zeros(n, dtype=float)

    # 5) BM25 vector for full corpus (zeros except candidates)
    bm25_score_vec = np.zeros(n, dtype=float)
    if len(bm25_map) > 0:
        # normalize BM25 across candidates for stability
        bm25_vals = np.array(list(bm25_map.values()), dtype=float)
        bm25_max = bm25_vals.max() if bm25_vals.size>0 else 0.0
        for idx, val in bm25_map.items():
            bm25_score_vec[idx] = float(val / (bm25_max + eps) if bm25_max > 0 else 0.0)
    # if no bm25 candidates, bm25_score_vec remains zeros

    # 6) Anchor normalization: map raw anchor scores (which may be cosine/inner-product distances) into [0,1]
    # we can normalize by the max returned anchor score to scale consistently
    if np.max(anchor_scores) > 0:
        anchor_max = float(np.max(anchor_scores))
        if anchor_max > 0:
            anchor_score_vec = anchor_score_vec / (anchor_max + eps)
    # else remain zeros

    # 7) Semantic normalization: optional - normalize semantic_score_vec by its max over the scored entries
    sem_max = semantic_score_vec.max() if semantic_score_vec.size > 0 else 0.0
    if sem_max > 0:
        semantic_score_vec = semantic_score_vec / (sem_max + eps)

    # 8) Final fusion
    final_scores = (
        alpha_anchor * anchor_score_vec +
        alpha_semantic * semantic_score_vec +
        alpha_bm25 * bm25_score_vec
    )

    # 9) Sort and return top_k results
    ranked_all = np.argsort(final_scores)[::-1]
    top_indices = ranked_all[:top_k]

    return df.iloc[top_indices].copy(), {
        "final_scores": final_scores,
        "anchor_scores": anchor_score_vec,
        "semantic_scores": semantic_score_vec,
        "bm25_scores": bm25_score_vec,
        "union_size": len(union_indices) if not full_semantic else n
    }
