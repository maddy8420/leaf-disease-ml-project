from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ================= LOAD MODEL =================
model = SentenceTransformer('all-MiniLM-L6-v2')

# ================= BUILD KNOWLEDGE =================

def build_knowledge(disease_info):
    knowledge = []

    # Core info
    knowledge.append(disease_info["cause"])

    t = disease_info["treatment"]["en"]
    knowledge.append(f"Use {t['medicine']} at {t['dosage']} {t['frequency']}")

    # Add optional knowledge
    if "knowledge" in disease_info:
        knowledge.extend(disease_info["knowledge"])

    return knowledge


# ================= SMART RETRIEVAL =================

def retrieve_best_match(query, knowledge):
    query_vec = model.encode([query])
    knowledge_vecs = model.encode(knowledge)

    scores = cosine_similarity(query_vec, knowledge_vecs)[0]
    best_idx = np.argmax(scores)

    return knowledge[best_idx], scores[best_idx]


# ================= RESPONSE ENGINE =================

def generate_response(query, context, disease_info):
    name = disease_info["name"].replace("_", " ")
    q = query.lower()

    # 🔥 HIGH LEVEL INTELLIGENCE

    if "disease" in q:
        return f"The detected disease is {name}."

    if "how many" in q or "times" in q:
        return f"You should follow this frequency: {disease_info['treatment']['en']['frequency']}."

    if "days" in q:
        return f"It should be applied {disease_info['treatment']['en']['frequency']}."

    if "enough" in q:
        return f"No, it should be repeated {disease_info['treatment']['en']['frequency']} for proper control."

    if "water" in q:
        return f"Avoid overwatering. {name} spreads in moisture-rich conditions."

    # 🔥 FALLBACK (SMART)
    return context


# ================= MAIN =================

def chatbot_reply(query, disease_info):
    knowledge = build_knowledge(disease_info)

    context, score = retrieve_best_match(query, knowledge)

    return generate_response(query, context, disease_info)