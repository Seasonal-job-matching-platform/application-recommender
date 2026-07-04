import pandas as pd
from app.database import SessionLocal
from app.embedding_service import EmbeddingService
from app.feature_engineering import (
    build_job_text, build_candidate_text, structured_features,
)


def rank_applicants(job_id: int, top_k: int = 5):

    session = SessionLocal()

    query = f"""
    SELECT
        j.id AS job_id,
        j.title,
        j.description,
        j.requirements,
        j.categories,
        j.location,
        j.work_arrangement,

        u.id AS user_id,
        u.country,
        u.fields_of_interest,

        r.skills,
        r.experience,
        r.education,
        r.languages,
        r.certificates,

        a.describe_yourself

    FROM applications a
    JOIN jobs j ON a.jobid = j.id
    JOIN users u ON a.userid = u.id
    LEFT JOIN resume r ON u.resume_id = r.id
    WHERE j.id = {job_id}
    """

    df = pd.read_sql(query, session.bind)
    session.close()

    if df.empty:
        return {"message": "No applicants found for this job."}

    embedder = EmbeddingService()

    # The job is identical for every applicant row — encode it ONCE.
    first = df.iloc[0]
    job = {
        "title": first["title"],
        "description": first["description"],
        "requirements": first["requirements"],
        "categories": first["categories"],
        "location": first["location"],
        "work_arrangement": first["work_arrangement"],
    }
    job_vec = embedder.encode(build_job_text(job))

    # Build every candidate's text, then encode them all in ONE batched call.
    candidate_texts = []
    for _, row in df.iterrows():
        resume = {
            "skills": row["skills"],
            "experience": row["experience"],
            "education": row["education"],
            "languages": row["languages"],
            "certificates": row["certificates"],
        }
        application = {"describe_yourself": row["describe_yourself"]}
        candidate_texts.append(build_candidate_text(resume, application))

    cand_vecs = embedder.encode_batch(candidate_texts)

    # Cheap loop: cosine + structured features, no more encoding.
    results = []
    for (_, row), cand_vec in zip(df.iterrows(), cand_vecs):
        semantic_sim = embedder.cosine_similarity(job_vec, cand_vec)

        user = {
            "country": row["country"],
            "fields_of_interest": row["fields_of_interest"],
        }
        resume = {"skills": row["skills"]}
        feats = structured_features(job, user, resume)

        score = (
            0.6 * semantic_sim +
            0.2 * feats["skills_jaccard"] +
            0.1 * feats["category_overlap"] +
            0.05 * feats["location_match"] +
            0.05 * min(feats["skills_overlap"], 5) / 5
        )

        candidate = row.to_dict()
        candidate["score"] = float(score)
        results.append(candidate)

    ranked = sorted(results, key=lambda x: x["score"], reverse=True)

    final_results = []
    for candidate in ranked[:top_k]:
        candidate_copy = candidate.copy()
        candidate_copy.pop("score", None)
        final_results.append(candidate_copy)

    return final_results
