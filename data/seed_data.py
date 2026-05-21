"""
Synthetic medical data for the MedX recommender prototype.
Simulates doctors, articles, and reading interactions on the coliquio platform.
"""

DOCTORS = [
    {"id": "d1",  "name": "Dr. Anna Müller",      "specialty": "cardiology",       "years_exp": 8},
    {"id": "d2",  "name": "Dr. Ben Schäfer",       "specialty": "neurology",        "years_exp": 12},
    {"id": "d3",  "name": "Dr. Clara Bauer",       "specialty": "oncology",         "years_exp": 5},
    {"id": "d4",  "name": "Dr. David König",       "specialty": "cardiology",       "years_exp": 15},
    {"id": "d5",  "name": "Dr. Eva Hoffmann",      "specialty": "pediatrics",       "years_exp": 7},
    {"id": "d6",  "name": "Dr. Felix Wagner",      "specialty": "neurology",        "years_exp": 10},
    {"id": "d7",  "name": "Dr. Greta Fischer",     "specialty": "oncology",         "years_exp": 3},
    {"id": "d8",  "name": "Dr. Hans Weber",        "specialty": "general_practice", "years_exp": 20},
    {"id": "d9",  "name": "Dr. Iris Meyer",        "specialty": "dermatology",      "years_exp": 6},
    {"id": "d10", "name": "Dr. Jonas Schulz",      "specialty": "psychiatry",       "years_exp": 9},
    {"id": "d11", "name": "Dr. Katharina Becker",  "specialty": "cardiology",       "years_exp": 11},
    {"id": "d12", "name": "Dr. Lars Richter",      "specialty": "radiology",        "years_exp": 14},
    {"id": "d13", "name": "Dr. Maria Klein",       "specialty": "pediatrics",       "years_exp": 4},
    {"id": "d14", "name": "Dr. Niklas Wolf",       "specialty": "general_practice", "years_exp": 18},
    {"id": "d15", "name": "Dr. Olivia Braun",      "specialty": "dermatology",      "years_exp": 2},
]

ARTICLES = [
    # Cardiology
    {"id": "a1",  "title": "New Guidelines for Heart Failure Management 2025",
     "tags": ["cardiology", "heart failure", "guidelines"],
     "specialty": "cardiology", "type": "guidelines",
     "summary": "Updated ESC guidelines for managing chronic and acute heart failure including SGLT2 inhibitors.",
     "complexity_score": 0.8, "reading_time_minutes": 14},
    {"id": "a2",  "title": "SGLT2 Inhibitors in Cardiovascular Disease: Clinical Evidence",
     "tags": ["cardiology", "pharmacology", "diabetes", "heart failure"],
     "specialty": "cardiology", "type": "clinical_evidence",
     "summary": "Comprehensive review of SGLT2 inhibitor trials and their cardiovascular outcomes.",
     "complexity_score": 0.85, "reading_time_minutes": 18},
    {"id": "a3",  "title": "Atrial Fibrillation: Anticoagulation Strategies in 2025",
     "tags": ["cardiology", "atrial fibrillation", "anticoagulation"],
     "specialty": "cardiology", "type": "review",
     "summary": "Current approaches to stroke prevention in AF including DOAC dosing and monitoring.",
     "complexity_score": 0.75, "reading_time_minutes": 12},
    {"id": "a4",  "title": "Hypertension Management: When to Escalate Therapy",
     "tags": ["cardiology", "hypertension", "pharmacology"],
     "specialty": "cardiology", "type": "clinical_guidance",
     "summary": "Practical guidance on resistant hypertension and combination therapy decisions.",
     "complexity_score": 0.5, "reading_time_minutes": 7},

    # Neurology
    {"id": "a5",  "title": "Migraine Prophylaxis: CGRP Antagonists in Practice",
     "tags": ["neurology", "migraine", "CGRP", "pharmacology"],
     "specialty": "neurology", "type": "clinical_evidence",
     "summary": "Real-world effectiveness of CGRP monoclonal antibodies in chronic migraine.",
     "complexity_score": 0.7, "reading_time_minutes": 10},
    {"id": "a6",  "title": "Stroke: Thrombolysis Window Extended — New Evidence",
     "tags": ["neurology", "stroke", "thrombolysis", "emergency"],
     "specialty": "neurology", "type": "research",
     "summary": "New imaging-guided trials support extending thrombolysis window in selected patients.",
     "complexity_score": 0.9, "reading_time_minutes": 20},
    {"id": "a7",  "title": "Multiple Sclerosis: Disease-Modifying Therapy Update",
     "tags": ["neurology", "multiple sclerosis", "immunology"],
     "specialty": "neurology", "type": "guidelines",
     "summary": "Latest DMT options and when to switch therapy in relapsing-remitting MS.",
     "complexity_score": 0.85, "reading_time_minutes": 16},
    {"id": "a8",  "title": "Parkinson's Disease: Non-Motor Symptoms Often Missed",
     "tags": ["neurology", "parkinson", "diagnosis"],
     "specialty": "neurology", "type": "education",
     "summary": "Recognizing and managing non-motor symptoms including sleep, mood, and autonomic issues.",
     "complexity_score": 0.45, "reading_time_minutes": 6},

    # Oncology
    {"id": "a9",  "title": "Immunotherapy in Non-Small Cell Lung Cancer: 2025 Review",
     "tags": ["oncology", "immunotherapy", "lung cancer", "PD-L1"],
     "specialty": "oncology", "type": "review",
     "summary": "PD-1/PD-L1 checkpoint inhibitors — patient selection and combination strategies.",
     "complexity_score": 0.9, "reading_time_minutes": 22},
    {"id": "a10", "title": "Breast Cancer Screening: Updated Recommendations",
     "tags": ["oncology", "breast cancer", "screening", "radiology"],
     "specialty": "oncology", "type": "guidelines",
     "summary": "New age and risk-based mammography screening recommendations from leading societies.",
     "complexity_score": 0.4, "reading_time_minutes": 5},
    {"id": "a11", "title": "CAR-T Cell Therapy: When and How to Refer",
     "tags": ["oncology", "immunotherapy", "hematology", "CAR-T"],
     "specialty": "oncology", "type": "clinical_guidance",
     "summary": "Practical guide to patient selection and referral pathways for CAR-T therapy.",
     "complexity_score": 0.8, "reading_time_minutes": 11},

    # Pediatrics
    {"id": "a12", "title": "RSV Prevention in Infants: Nirsevimab Evidence",
     "tags": ["pediatrics", "RSV", "vaccination", "neonatal"],
     "specialty": "pediatrics", "type": "clinical_evidence",
     "summary": "Phase 3 data on nirsevimab for RSV prophylaxis in all infants entering first RSV season.",
     "complexity_score": 0.6, "reading_time_minutes": 8},
    {"id": "a13", "title": "ADHD Diagnosis in Children: Updated DSM-5 Criteria",
     "tags": ["pediatrics", "ADHD", "psychiatry", "diagnosis"],
     "specialty": "pediatrics", "type": "education",
     "summary": "Practical diagnostic approach and common pitfalls in childhood ADHD assessment.",
     "complexity_score": 0.35, "reading_time_minutes": 5},

    # Dermatology
    {"id": "a14", "title": "Atopic Dermatitis: Biologic Therapies Beyond Dupilumab",
     "tags": ["dermatology", "atopic dermatitis", "biologics", "immunology"],
     "specialty": "dermatology", "type": "review",
     "summary": "Tralokinumab, lebrikizumab, and JAK inhibitors — positioning in moderate-to-severe AD.",
     "complexity_score": 0.8, "reading_time_minutes": 13},
    {"id": "a15", "title": "Melanoma Early Detection: Dermoscopy Patterns to Know",
     "tags": ["dermatology", "melanoma", "dermoscopy", "oncology"],
     "specialty": "dermatology", "type": "education",
     "summary": "Key dermoscopic criteria and the ABCDE rule updated for 2025 practice.",
     "complexity_score": 0.3, "reading_time_minutes": 4},

    # General Practice / Cross-specialty
    {"id": "a16", "title": "Antibiotic Stewardship in Primary Care: Practical Guide",
     "tags": ["general_practice", "antibiotics", "stewardship", "infection"],
     "specialty": "general_practice", "type": "clinical_guidance",
     "summary": "When to prescribe, when to wait, and how to communicate with patients about antibiotics.",
     "complexity_score": 0.3, "reading_time_minutes": 4},
    {"id": "a17", "title": "Diabetes Type 2: New Individualized Treatment Algorithm",
     "tags": ["general_practice", "diabetes", "cardiology", "endocrinology"],
     "specialty": "general_practice", "type": "guidelines",
     "summary": "ADA/EASD 2025 consensus: cardiovascular and renal risk as the starting point for therapy.",
     "complexity_score": 0.55, "reading_time_minutes": 8},
    {"id": "a18", "title": "Long COVID: Symptoms, Diagnosis and Management",
     "tags": ["general_practice", "COVID-19", "post-COVID", "rehabilitation"],
     "specialty": "general_practice", "type": "review",
     "summary": "Evidence-based approach to diagnosing and managing post-acute sequelae of SARS-CoV-2.",
     "complexity_score": 0.4, "reading_time_minutes": 6},

    # Psychiatry
    {"id": "a19", "title": "Treatment-Resistant Depression: Ketamine and Esketamine",
     "tags": ["psychiatry", "depression", "ketamine", "pharmacology"],
     "specialty": "psychiatry", "type": "clinical_evidence",
     "summary": "Clinical outcomes with IV ketamine and intranasal esketamine in TRD patients.",
     "complexity_score": 0.75, "reading_time_minutes": 11},
    {"id": "a20", "title": "Burnout in Physicians: Recognition and Intervention",
     "tags": ["psychiatry", "burnout", "physician_health", "wellbeing"],
     "specialty": "psychiatry", "type": "education",
     "summary": "Screening tools, risk factors, and evidence-based interventions for physician burnout.",
     "complexity_score": 0.2, "reading_time_minutes": 4},

    # Radiology
    {"id": "a21", "title": "AI in Radiology: Current Clinical Applications",
     "tags": ["radiology", "AI", "machine learning", "imaging"],
     "specialty": "radiology", "type": "review",
     "summary": "FDA-cleared AI tools in chest X-ray, CT, and mammography — practical overview.",
     "complexity_score": 0.6, "reading_time_minutes": 9},
    {"id": "a22", "title": "Low-Dose CT for Lung Cancer Screening: Who Qualifies?",
     "tags": ["radiology", "lung cancer", "screening", "oncology"],
     "specialty": "radiology", "type": "guidelines",
     "summary": "Updated LDCT screening eligibility criteria and reporting standards (Lung-RADS 2025).",
     "complexity_score": 0.5, "reading_time_minutes": 7},

    # Additional cardiology
    {"id": "a23", "title": "Statin Therapy in Primary Prevention: Who Benefits in 2025?",
     "tags": ["cardiology", "lipids", "statins", "prevention"],
     "specialty": "cardiology", "type": "guidelines",
     "summary": "Updated risk-based criteria for initiating statins in primary cardiovascular prevention.",
     "complexity_score": 0.65, "reading_time_minutes": 9},
    {"id": "a24", "title": "TAVR vs Surgical AVR: Quick Decision Guide",
     "tags": ["cardiology", "TAVR", "valve disease", "intervention"],
     "specialty": "cardiology", "type": "clinical_guidance",
     "summary": "When to refer for transcatheter aortic valve replacement versus open surgery.",
     "complexity_score": 0.55, "reading_time_minutes": 6},

    # Additional neurology
    {"id": "a25", "title": "First-Line Antiseizure Drugs: Practical Selection",
     "tags": ["neurology", "epilepsy", "pharmacology"],
     "specialty": "neurology", "type": "clinical_guidance",
     "summary": "Choosing initial antiseizure therapy based on seizure type, age, and comorbidities.",
     "complexity_score": 0.6, "reading_time_minutes": 7},
    {"id": "a26", "title": "Early Signs of Dementia GPs Should Not Miss",
     "tags": ["neurology", "dementia", "diagnosis", "general_practice"],
     "specialty": "neurology", "type": "education",
     "summary": "Subtle cognitive and functional red flags for timely referral and workup.",
     "complexity_score": 0.35, "reading_time_minutes": 5},

    # Additional oncology
    {"id": "a27", "title": "Colorectal Cancer Screening: FIT vs Colonoscopy",
     "tags": ["oncology", "colorectal", "screening", "prevention"],
     "specialty": "oncology", "type": "guidelines",
     "summary": "Comparing stool-based screening intervals and colonoscopy pathways by risk group.",
     "complexity_score": 0.45, "reading_time_minutes": 6},
    {"id": "a28", "title": "Managing Immune-Related Adverse Events from Checkpoint Inhibitors",
     "tags": ["oncology", "immunotherapy", "toxicity", "emergency"],
     "specialty": "oncology", "type": "clinical_guidance",
     "summary": "Recognition and first-line management of common irAEs in outpatient and acute settings.",
     "complexity_score": 0.85, "reading_time_minutes": 15},

    # Additional pediatrics & general practice
    {"id": "a29", "title": "Childhood Obesity: First-Line Lifestyle and When to Refer",
     "tags": ["pediatrics", "obesity", "lifestyle", "metabolic"],
     "specialty": "pediatrics", "type": "clinical_guidance",
     "summary": "Structured lifestyle counselling, screening labs, and referral thresholds in children.",
     "complexity_score": 0.4, "reading_time_minutes": 5},
    {"id": "a30", "title": "Vitamin D Deficiency in Primary Care: Test or Treat?",
     "tags": ["general_practice", "vitamin D", "supplementation", "bone health"],
     "specialty": "general_practice", "type": "review",
     "summary": "When to measure 25-OH vitamin D and evidence-based supplementation strategies.",
     "complexity_score": 0.3, "reading_time_minutes": 4},

    # Additional dermatology & psychiatry
    {"id": "a31", "title": "Plaque Psoriasis: Biologic Options in 5 Minutes",
     "tags": ["dermatology", "psoriasis", "biologics", "immunology"],
     "specialty": "dermatology", "type": "education",
     "summary": "Quick comparison of IL-17, IL-23, and TNF inhibitors for moderate plaque psoriasis.",
     "complexity_score": 0.5, "reading_time_minutes": 5},
    {"id": "a32", "title": "Generalised Anxiety Disorder: SSRI Selection and Monitoring",
     "tags": ["psychiatry", "anxiety", "SSRI", "pharmacology"],
     "specialty": "psychiatry", "type": "clinical_guidance",
     "summary": "First-line SSRI choices, titration, and follow-up for GAD in outpatient practice.",
     "complexity_score": 0.55, "reading_time_minutes": 7},
]

# Reading interactions: (doctor_id, article_id, rating 1-5)
INTERACTIONS = [
    # Cardiologists read cardiology + some cross-specialty
    ("d1",  "a1",  5), ("d1",  "a2",  4), ("d1",  "a3",  5), ("d1",  "a17", 3),
    ("d4",  "a1",  4), ("d4",  "a2",  5), ("d4",  "a4",  5), ("d4",  "a3",  3),
    ("d11", "a1",  5), ("d11", "a3",  4), ("d11", "a4",  4), ("d11", "a2",  3),

    # Neurologists
    ("d2",  "a5",  5), ("d2",  "a6",  4), ("d2",  "a7",  5), ("d2",  "a8",  4),
    ("d6",  "a5",  4), ("d6",  "a6",  5), ("d6",  "a7",  4), ("d6",  "a8",  3),

    # Oncologists
    ("d3",  "a9",  5), ("d3",  "a10", 4), ("d3",  "a11", 5),
    ("d7",  "a9",  4), ("d7",  "a10", 5), ("d7",  "a11", 3),

    # Pediatricians
    ("d5",  "a12", 5), ("d5",  "a13", 4), ("d5",  "a18", 3),
    ("d13", "a12", 4), ("d13", "a13", 5),

    # Dermatologists
    ("d9",  "a14", 5), ("d9",  "a15", 4),
    ("d15", "a14", 4), ("d15", "a15", 5),

    # General practitioners read broadly
    ("d8",  "a16", 5), ("d8",  "a17", 5), ("d8",  "a18", 4), ("d8",  "a1",  3),
    ("d14", "a16", 4), ("d14", "a17", 4), ("d14", "a18", 5), ("d14", "a12", 3),

    # Psychiatrists
    ("d10", "a19", 5), ("d10", "a20", 5), ("d10", "a13", 3),

    # Radiologist
    ("d12", "a21", 5), ("d12", "a22", 5), ("d12", "a10", 4),

    # Cross-specialty reads (adds collaborative signal)
    ("d1",  "a18", 3), ("d2",  "a20", 3), ("d4",  "a9",  3),
    ("d6",  "a20", 4), ("d8",  "a5",  3), ("d9",  "a10", 4),
    ("d11", "a17", 4), ("d12", "a9",  3), ("d3",  "a22", 4),

    # New articles — specialty reads
    ("d1",  "a23", 4), ("d4",  "a23", 5), ("d11", "a24", 4),
    ("d2",  "a25", 4), ("d6",  "a25", 5), ("d8",  "a26", 4),
    ("d3",  "a27", 4), ("d7",  "a28", 3), ("d5",  "a29", 5),
    ("d13", "a29", 4), ("d8",  "a30", 5), ("d14", "a30", 4),
    ("d9",  "a31", 4), ("d15", "a31", 5), ("d10", "a32", 4),
    ("d12", "a28", 3), ("d1",  "a24", 3), ("d14", "a26", 4),
]
