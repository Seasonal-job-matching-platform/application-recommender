# Applicant Recommender — Performance Fix Spec

This document describes all changes needed to fix the `application-recommender` service.
Hand this to an implementation session. **Repo:** `application-recommender-main`.

---

## Problem being solved

The service (recruiter side: rank the applicants who applied to a given job) is slow and
intermittently returns nothing. Two independent causes:

1. **Cold start (per reopen):** `EmbeddingService` downloads `all-mpnet-base-v2` (~420 MB) +
   a CrossEncoder from HuggingFace **on the first request** after every fresh start. The
   HF cache is not under Azure's persistent `/home`, so every stop/reopen re-downloads.
   The first request takes ~71 s, blowing the backend's 25 s read timeout → the recruiter
   UI shows "No applicant recommendations available."

2. **Warm requests are also slow:** even after the model is loaded, each request runs ~14
   sequential neural encodes for only 7 applicants — the job text is re-encoded once per
   applicant, and every candidate is encoded one-at-a-time in a Python loop with a heavy
   768-dim model.

## Strategy (agreed)

- **Tier 3 — smaller model:** swap `all-mpnet-base-v2` → `all-MiniLM-L6-v2` (384-dim, ~90 MB,
  ~5× faster on CPU). A slightly smaller model is acceptable. Bonus: MiniLM's weight file is
  ~90 MB (under GitHub's 100 MB/file limit), so it can be **bundled in the repo** — no runtime
  download, ever. This is the same model the jobs matching engine already uses.
- **Tier 1 — encode once + batch:** encode the job text **once**, encode all candidate texts
  in a **single batched call**, then do cosine + structured features in a cheap loop.
- **Cold-start hardening:** bundle the model locally + warm it up at startup.
- **Cross-encoder:** make it **lazy** (only the unused `/recommend/talent` path needs it), so
  the applicants path never loads it.

Ranking weights and output shape stay identical. No pgvector / no database schema changes.

---

## Summary of file changes

| File | Change |
|---|---|
| `app/model_cache/` (new dir) | Bundle `all-MiniLM-L6-v2` weights |
| `app/embedding_service.py` | Load MiniLM from local dir; add `encode_batch`; lazy cross-encoder; normalized-vector cosine |
| `app/feature_engineering.py` | Split into `build_job_text` / `build_candidate_text` / `structured_features` (no encoding inside) |
| `app/services/applicant_ranking.py` | Encode job once, batch-encode candidates, combine scores |
| `app/main.py` | Add startup warmup (FastAPI `lifespan`) |
| `.gitignore` | Ensure `app/model_cache/` is NOT ignored |
| Backend `HttpClientConfig.java` (other repo) | Optional: read timeout 25s → 60s |

---

## 1. Bundle the model — `app/model_cache/`

Create `app/model_cache/` containing the `all-MiniLM-L6-v2` weights. Two options:

**Option A (recommended):** copy the already-bundled model from the jobs matching engine repo,
which ships it at `app/model_cache/` (files: `config.json`, `model.safetensors`,
`tokenizer.json`, `vocab.txt`, `modules.json`, `1_Pooling/config.json`, etc.).

**Option B:** download it once with a script and commit the result:
```python
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2").save("app/model_cache")
```

Then confirm `app/model_cache/` is committed (it's ~90 MB — under GitHub's per-file limit, so a
normal commit works; **no Git LFS needed**). Make sure `.gitignore` does not exclude it.

---

## 2. `app/embedding_service.py` (full replacement)

```python
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder

_LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "model_cache"
_BI_ENCODER_NAME = "all-MiniLM-L6-v2"
_CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class EmbeddingService:

    _bi_encoder = None
    _cross_encoder = None

    def __init__(self):
        if EmbeddingService._bi_encoder is None:
            if _LOCAL_MODEL_DIR.exists() and any(_LOCAL_MODEL_DIR.iterdir()):
                EmbeddingService._bi_encoder = SentenceTransformer(str(_LOCAL_MODEL_DIR))
            else:
                # Fallback only if the bundled model is missing
                EmbeddingService._bi_encoder = SentenceTransformer(_BI_ENCODER_NAME)
        self.bi_encoder = EmbeddingService._bi_encoder

    @property
    def cross_encoder(self):
        # Loaded lazily — only the /recommend/talent path uses it.
        if EmbeddingService._cross_encoder is None:
            EmbeddingService._cross_encoder = CrossEncoder(_CROSS_ENCODER_NAME)
        return EmbeddingService._cross_encoder

    def encode(self, text: str):
        return self.bi_encoder.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )

    def encode_batch(self, texts: list[str]):
        return self.bi_encoder.encode(
            texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True
        )

    def cosine_similarity(self, vec1, vec2):
        # Vectors are L2-normalized, so dot product == cosine similarity.
        return float(np.dot(vec1, vec2))

    def cross_score(self, pairs):
        return self.cross_encoder.predict(pairs)
```

Notes:
- `cross_encoder` is now a lazy `@property`. `talent_ranking.py` calls `self.embedder.cross_score(...)`,
  which still works unchanged.
- Embeddings are L2-normalized, so `cosine_similarity` is a dot product (faster, same result).

---

## 3. `app/feature_engineering.py`

Keep `clean_text`, `normalize_list`, `jaccard_similarity`, `overlap_count` as they are.
**Delete** `build_feature_vector` and replace it with the three functions below (they no longer
do any encoding — encoding is moved up into the ranker so it can be batched).

```python
def build_job_text(job):
    job_requirements = normalize_list(job.get("requirements"))
    job_categories = normalize_list(job.get("categories"))
    return " ".join([
        clean_text(job.get("title")),
        clean_text(job.get("description")),
        " ".join(job_requirements),
        " ".join(job_categories),
    ])


def build_candidate_text(resume, application):
    resume_skills = normalize_list(resume.get("skills"))
    resume_experience = normalize_list(resume.get("experience"))
    resume_education = normalize_list(resume.get("education"))
    resume_certificates = normalize_list(resume.get("certificates"))
    return " ".join([
        " ".join(resume_skills),
        " ".join(resume_experience),
        " ".join(resume_education),
        " ".join(resume_certificates),
        clean_text(application.get("describe_yourself")),
    ])


def structured_features(job, user, resume):
    job_requirements = normalize_list(job.get("requirements"))
    job_categories = normalize_list(job.get("categories"))
    user_interests = normalize_list(user.get("fields_of_interest"))
    resume_skills = normalize_list(resume.get("skills"))

    skills_jaccard = jaccard_similarity(job_requirements, resume_skills)
    skills_overlap = overlap_count(job_requirements, resume_skills)
    category_overlap = jaccard_similarity(job_categories, user_interests)

    # NOTE: use (x or "") — job.get("work_arrangement", "") returns None if the
    # column exists but is NULL, which would crash .upper().
    location_match = 0
    if (job.get("work_arrangement") or "").upper() == "REMOTE":
        location_match = 1
    elif job.get("location") == user.get("country"):
        location_match = 1

    return {
        "skills_jaccard": skills_jaccard,
        "skills_overlap": skills_overlap,
        "category_overlap": category_overlap,
        "location_match": location_match,
    }
```

---

## 4. `app/services/applicant_ranking.py` (full replacement)

Encode the job **once**, batch-encode all candidates in **one** call, then loop cheaply.

```python
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
        c = candidate.copy()
        c.pop("score", None)
        final_results.append(c)

    return final_results
```

Ranking weights (0.6 / 0.2 / 0.1 / 0.05 / 0.05) and the score-stripped output are unchanged.

---

## 5. `app/main.py` — startup warmup

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.recommender import recommend_applicants, recommend_talent
from app.embedding_service import EmbeddingService


@asynccontextmanager
async def lifespan(app):
    # Load the bi-encoder into memory BEFORE serving traffic, so the first
    # real request is warm instead of paying the model-load cost.
    EmbeddingService().encode("warmup")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/recommend/applicants/{job_id}")
def get_applicant_recommendations(job_id: int, top_k: int = 5):
    return recommend_applicants(job_id, top_k)


@app.get("/recommend/talent/{job_id}")
def get_talent_recommendations(job_id: int, top_k: int = 5):
    return recommend_talent(job_id, top_k)
```

Uvicorn will not accept requests until `lifespan` startup completes, so the warmup guarantees
the model is resident before the first request.

---

## 6. Optional / secondary

- **Backend read timeout** (`HttpClientConfig.java`, the Spring backend repo): with the model
  bundled + warmed + MiniLM, warm requests should be ~1 s, so the existing 25 s is fine. Raising
  `setReadTimeout(25000)` → `60000` is only worth it as insurance for the very first request after
  a brand-new deploy. Low priority.
- **Talent path** (`talent_ranking.py`): it automatically benefits from the MiniLM swap. It still
  builds its candidate embedding cache one-at-a-time; it could use `encode_batch` too, and its
  CrossEncoder still downloads from HF on first use. Since the backend never calls
  `/recommend/talent`, leave this unless/until that endpoint is used.
- **`requirements.txt`:** no change required (`sentence-transformers` already covers both encoders).
  `all-mpnet-base-v2` was never a dependency name, just a downloaded model, so nothing to remove.

---

## 7. Verification

1. **Local, cold:** delete any HF cache, start the app. Confirm startup logs show the model
   loading from `app/model_cache` (not "sending requests to the HF Hub"), and that startup
   completes before the server is ready.
2. **Warm latency:** call `GET /recommend/applicants/<job_id>` twice. Both should return in
   ~1–2 s for a handful of applicants (previously ~15 s warm / ~71 s cold).
3. **Correctness:** results are a non-empty list ordered by relevance; the same strong applicants
   still rank at the top (order may shift slightly vs mpnet — expected with a smaller model).
4. **Cross-encoder not loaded:** confirm the applicants path never logs a CrossEncoder / HF
   download. Only `/recommend/talent` should trigger it.
5. **Azure reopen:** stop and restart the app. After the container finishes booting (~1–2 min of
   Azure overhead, unrelated to the model), the first applicants request should be fast — no
   71 s download, no backend timeout.
```
