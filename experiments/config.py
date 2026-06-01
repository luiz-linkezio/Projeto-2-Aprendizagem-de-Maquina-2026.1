from __future__ import annotations

SEED = 42
TEST_SIZE = 0.30
OPTUNA_N_TRIALS = 10
OPTUNA_CV_FOLDS = 2
AUTOGLUON_TIME_DEFAULT = 3_600      # 1h
AUTOGLUON_TIME_EXTREME = 2 * 3_600  # 2h (reduzido de 4h por limitação de RAM)

DATASETS: list[dict] = [
    # ── Pequeno (n < 1 000) ── apenas 3 disponíveis no TabArena-v0.1
    # Nota: o benchmark não oferece 10 datasets com n < 1.000 para classificação.
    {"task_id": 363621, "name": "blood-transfusion-service-center",         "n": 748,    "regime": "pequeno", "tipo": "binario"},
    {"task_id": 363629, "name": "diabetes",                                 "n": 768,    "regime": "pequeno", "tipo": "binario"},
    {"task_id": 363614, "name": "anneal",                                   "n": 898,    "regime": "pequeno", "tipo": "multiclasse"},
    # ── Médio (1 000 ≤ n ≤ 10 000) ─────────────────────────────────────────
    {"task_id": 363626, "name": "credit-g",                                 "n": 1_000,  "regime": "medio",   "tipo": "binario"},
    {"task_id": 363685, "name": "maternal_health_risk",                     "n": 1_014,  "regime": "medio",   "tipo": "multiclasse"},
    {"task_id": 363696, "name": "qsar-biodeg",                              "n": 1_054,  "regime": "medio",   "tipo": "binario"},
    {"task_id": 363707, "name": "website_phishing",                         "n": 1_353,  "regime": "medio",   "tipo": "multiclasse"},
    {"task_id": 363671, "name": "Fitness_Club",                             "n": 1_500,  "regime": "medio",   "tipo": "binario"},
    {"task_id": 363711, "name": "MIC",                                      "n": 1_699,  "regime": "medio",   "tipo": "multiclasse"},
    {"task_id": 363682, "name": "Is-this-a-good-customer",                  "n": 1_723,  "regime": "medio",   "tipo": "binario"},
    {"task_id": 363684, "name": "Marketing_Campaign",                       "n": 2_240,  "regime": "medio",   "tipo": "binario"},
    {"task_id": 363700, "name": "seismic-bumps",                            "n": 2_584,  "regime": "medio",   "tipo": "binario"},
    {"task_id": 363702, "name": "splice",                                   "n": 3_190,  "regime": "medio",   "tipo": "multiclasse"},
    {"task_id": 363704, "name": "students_dropout_and_academic_success",    "n": 4_424,  "regime": "medio",   "tipo": "multiclasse"},
    {"task_id": 363623, "name": "churn",                                    "n": 5_000,  "regime": "medio",   "tipo": "binario"},
    {"task_id": 363694, "name": "polish_companies_bankruptcy",              "n": 5_910,  "regime": "medio",   "tipo": "binario"},
    {"task_id": 363706, "name": "taiwanese_bankruptcy_prediction",          "n": 6_819,  "regime": "medio",   "tipo": "binario"},
    # ── Grande (n > 10 000) ──────────────────────────────────────────────────
    {"task_id": 363676, "name": "heloc",                                    "n": 10_459, "regime": "grande",  "tipo": "binario"},
    {"task_id": 363712, "name": "jm1",                                      "n": 10_885, "regime": "grande",  "tipo": "binario"},
    {"task_id": 363632, "name": "E-CommereShippingData",                    "n": 10_999, "regime": "grande",  "tipo": "binario"},
    {"task_id": 363691, "name": "online_shoppers_intention",                "n": 12_330, "regime": "grande",  "tipo": "binario"},
    {"task_id": 363681, "name": "in_vehicle_coupon_recommendation",         "n": 12_684, "regime": "grande",  "tipo": "binario"},
    {"task_id": 363679, "name": "HR_Analytics_Job_Change_of_Data_Scientists","n": 19_158,"regime": "grande",  "tipo": "binario"},
    {"task_id": 363627, "name": "credit_card_clients_default",              "n": 30_000, "regime": "grande",  "tipo": "binario"},
    {"task_id": 363613, "name": "Amazon_employee_access",                   "n": 32_769, "regime": "grande",  "tipo": "binario"},
    {"task_id": 363618, "name": "bank-marketing",                           "n": 45_211, "regime": "grande",  "tipo": "binario"},
    {"task_id": 363616, "name": "APSFailure",                               "n": 76_000, "regime": "grande",  "tipo": "binario"},
    {"task_id": 363630, "name": "Diabetes130US",                            "n": 71_518, "regime": "grande",  "tipo": "binario"},
    {"task_id": 363699, "name": "SDSS17",                                   "n": 78_053, "regime": "grande",  "tipo": "multiclasse"},
    {"task_id": 363628, "name": "customer_satisfaction_in_airline",         "n": 129_880,"regime": "grande",  "tipo": "binario"},
]

MODELS = ["mambular", "lightgbm", "catboost", "xgboost", "autogluon_default", "autogluon_extreme"]
