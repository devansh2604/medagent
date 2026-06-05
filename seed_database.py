"""
seed_database.py — Seeds ChromaDB with 200 realistic synthetic patient records.
Run independently: python3 seed_database.py
"""

import uuid
import random
import rag

# ── Seed data pools ───────────────────────────────────────────────────────────

FIRST_NAMES_M = ["James","Robert","John","Michael","David","William","Richard","Joseph",
                  "Thomas","Charles","Arjun","Rahul","Mohammed","Omar","Carlos","Luis",
                  "Ahmed","Daniel","Kevin","Brian","George","Jason","Timothy","Ronald"]
FIRST_NAMES_F = ["Mary","Patricia","Jennifer","Linda","Barbara","Elizabeth","Susan","Jessica",
                  "Sarah","Karen","Priya","Anjali","Fatima","Nadia","Maria","Sofia","Aisha",
                  "Amanda","Melissa","Stephanie","Rebecca","Sharon","Laura","Rachel"]
LAST_NAMES    = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
                  "Wilson","Anderson","Taylor","Thomas","Hernandez","Moore","Martin","Jackson",
                  "Thompson","White","Lopez","Lee","Patel","Kumar","Sharma","Khan","Chen"]

DIAGNOSES = [
    {
        "name": "Type 2 Diabetes",
        "symptoms": "increased thirst, frequent urination, blurred vision, fatigue, slow-healing sores",
        "meds": "Metformin 500mg, Glipizide 5mg",
        "history": "Obesity, family history of diabetes",
    },
    {
        "name": "Hypertension",
        "symptoms": "headache, dizziness, shortness of breath, chest tightness",
        "meds": "Lisinopril 10mg, Amlodipine 5mg",
        "history": "High salt diet, sedentary lifestyle",
    },
    {
        "name": "Asthma",
        "symptoms": "wheezing, shortness of breath, chest tightness, chronic cough",
        "meds": "Albuterol inhaler, Fluticasone inhaler",
        "history": "Childhood asthma, dust mite allergy",
    },
    {
        "name": "GERD",
        "symptoms": "heartburn, acid reflux, regurgitation, chest pain after meals, difficulty swallowing",
        "meds": "Omeprazole 20mg, Antacids PRN",
        "history": "Obesity, frequent fatty food intake, caffeine excess",
    },
    {
        "name": "Viral Fever",
        "symptoms": "high fever, body aches, fatigue, runny nose, sore throat, chills",
        "meds": "Paracetamol 500mg, rest and hydration",
        "history": "Recent exposure to flu season, no vaccination",
    },
    {
        "name": "Depression",
        "symptoms": "persistent sadness, loss of interest, fatigue, sleep disturbances, appetite changes, hopelessness",
        "meds": "Sertraline 50mg, Cognitive Behavioral Therapy",
        "history": "Previous depressive episode, family history of mood disorders",
    },
    {
        "name": "Anxiety Disorder",
        "symptoms": "excessive worry, restlessness, palpitations, sweating, insomnia, muscle tension",
        "meds": "Escitalopram 10mg, Lorazepam 0.5mg PRN",
        "history": "Work-related stress, perfectionist personality, childhood trauma",
    },
    {
        "name": "Migraine",
        "symptoms": "severe unilateral headache, nausea, vomiting, photophobia, phonophobia, visual aura",
        "meds": "Sumatriptan 50mg, Propranolol 40mg preventive",
        "history": "Family history of migraines, hormonal triggers, stress",
    },
    {
        "name": "Rheumatoid Arthritis",
        "symptoms": "joint pain, morning stiffness, swollen joints, fatigue, low-grade fever",
        "meds": "Methotrexate 15mg weekly, Prednisolone 5mg",
        "history": "Autoimmune family history, female predominance",
    },
    {
        "name": "Hypothyroidism",
        "symptoms": "fatigue, weight gain, cold intolerance, constipation, dry skin, hair loss, brain fog",
        "meds": "Levothyroxine 50mcg daily",
        "history": "Hashimoto's thyroiditis, family history of thyroid disease",
    },
    {
        "name": "Hyperthyroidism",
        "symptoms": "weight loss, heat intolerance, palpitations, tremors, anxiety, frequent bowel movements",
        "meds": "Methimazole 10mg, Propranolol 20mg",
        "history": "Graves disease, family history of autoimmune conditions",
    },
    {
        "name": "Chronic Kidney Disease",
        "symptoms": "fatigue, swelling in legs, decreased urine output, nausea, shortness of breath",
        "meds": "Furosemide 40mg, Erythropoietin injections, phosphate binders",
        "history": "Long-standing diabetes, hypertension",
    },
    {
        "name": "Coronary Artery Disease",
        "symptoms": "chest pain on exertion, shortness of breath, fatigue, palpitations",
        "meds": "Aspirin 81mg, Atorvastatin 40mg, Metoprolol 25mg",
        "history": "Smoking, hyperlipidemia, family history of heart disease",
    },
    {
        "name": "COPD",
        "symptoms": "chronic productive cough, progressive breathlessness, wheezing, frequent respiratory infections",
        "meds": "Tiotropium inhaler, Formoterol inhaler, Prednisolone during exacerbations",
        "history": "40 pack-year smoking history, occupational dust exposure",
    },
    {
        "name": "Iron Deficiency Anemia",
        "symptoms": "fatigue, pallor, shortness of breath, cold hands, brittle nails, dizziness",
        "meds": "Ferrous sulfate 325mg, dietary iron supplementation",
        "history": "Poor diet, heavy menstrual bleeding, chronic GI blood loss",
    },
]

ALLERGIES_POOL = [
    "Penicillin", "Sulfa drugs", "Aspirin", "NSAIDs", "Shellfish",
    "Peanuts", "Latex", "Contrast dye", "No known allergies", "Codeine",
    "No known allergies", "No known allergies",  # weighted toward no allergy
]

GENDERS = ["Male", "Female"]


def random_patient() -> dict:
    gender = random.choice(GENDERS)
    if gender == "Male":
        first = random.choice(FIRST_NAMES_M)
    else:
        first = random.choice(FIRST_NAMES_F)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"

    age = random.randint(18, 80)
    dx  = random.choice(DIAGNOSES)

    # Add mild variation to symptoms
    extra_symptoms = random.choice([
        "", ", fatigue", ", loss of appetite", ", insomnia",
        ", mild fever", ", weight loss", ", joint pain",
    ])

    return {
        "name":            name,
        "age":             age,
        "gender":          gender,
        "symptoms":        dx["symptoms"] + extra_symptoms,
        "medical_history": dx["history"],
        "medications":     dx["meds"],
        "allergies":       random.choice(ALLERGIES_POOL),
        "diagnosis":       dx["name"],
    }


def seed_database(n: int = 200) -> None:
    print(f"Seeding ChromaDB with {n} synthetic patient records...")
    existing = rag.get_stats()["total_records"]
    if existing >= n:
        print(f"  ✓ Already have {existing} records. Skipping seed.")
        return

    to_add = n - existing
    print(f"  Adding {to_add} records (already have {existing})...")

    for i in range(to_add):
        pid    = f"SEED-{uuid.uuid4().hex[:8].upper()}"
        record = random_patient()
        record["patient_id"] = pid
        rag.upsert_patient(pid, record)

        if (i + 1) % 50 == 0:
            print(f"  → {i + 1}/{to_add} inserted...")

    final = rag.get_stats()["total_records"]
    print(f"  ✓ Seed complete. Total records in ChromaDB: {final}")


if __name__ == "__main__":
    seed_database(200)
