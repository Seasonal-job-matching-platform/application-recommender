import numpy as np
import re


# ----------------------------
# Text Normalization Utilities
# ----------------------------

def clean_text(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.lower().strip())


def normalize_list(field):
    """
    - Splits comma-separated values
    - Lowercases
    - Removes extra spaces
    """
    if not field:
        return []

    normalized = []

    for item in field:
        if not item:
            continue

        if isinstance(item, str):
            parts = item.split(",")
            for p in parts:
                cleaned = clean_text(p)
                if cleaned:
                    normalized.append(cleaned)
        else:
            normalized.append(str(item).lower())

    return normalized


# ----------------------------
# Similarity Metrics
# ----------------------------

def jaccard_similarity(list1, list2):
    if not list1 or not list2:
        return 0.0

    set1 = set(list1)
    set2 = set(list2)

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def overlap_count(list1, list2):
    if not list1 or not list2:
        return 0
    return len(set(list1) & set(list2))


# ----------------------------
# Text Builders (no encoding — encoding happens in the ranker so it can be batched)
# ----------------------------

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

    # NOTE: (x or "") because work_arrangement can be NULL in the database,
    # which .get(..., "") does not protect against and would crash .upper().
    location_match = 0
    if (job.get("work_arrangement") or "").upper() == "REMOTE":
        location_match = 1  # Remote jobs ignore country mismatch
    elif job.get("location") == user.get("country"):
        location_match = 1

    return {
        "skills_jaccard": skills_jaccard,
        "skills_overlap": skills_overlap,
        "category_overlap": category_overlap,
        "location_match": location_match,
    }


# ----------------------------
# Feature Builder (used by training/train_model.py for the XGBoost ranker)
# ----------------------------

def build_feature_vector(job, user, resume, application, embedding_service):

    job_vec = embedding_service.encode(build_job_text(job))
    cand_vec = embedding_service.encode(build_candidate_text(resume, application))
    semantic_sim = embedding_service.cosine_similarity(job_vec, cand_vec)

    feats = structured_features(job, user, resume)

    # ---- Richness signals ----
    num_skills = len(normalize_list(resume.get("skills")))
    num_experience = len(normalize_list(resume.get("experience")))
    num_education = len(normalize_list(resume.get("education")))
    num_languages = len(normalize_list(resume.get("languages")))
    num_certificates = len(normalize_list(resume.get("certificates")))

    # ---- Final feature vector ----
    return np.array([
        semantic_sim,                # 0
        feats["skills_jaccard"],     # 1
        feats["skills_overlap"],     # 2
        feats["category_overlap"],   # 3
        feats["location_match"],     # 4
        num_skills,                  # 5
        num_experience,              # 6
        num_education,               # 7
        num_languages,               # 8
        num_certificates             # 9
    ])