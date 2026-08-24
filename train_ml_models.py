# train_ml_models.py
#
# Clean, reproducible ML + XAI pipeline for the nutrition deficiency thesis.
# Replaces the DeepSeek-generated Food_Nutrition_deficit.ipynb, which had
# several real, execution-verified problems (see chat history / thesis notes
# for the full diagnostic):
#   1. Not reproducible end-to-end: a NameError crash (Cell 8), 8 never-run
#      cells (9-16), a SHAP crash (Cell 23), and a `df` whose schema silently
#      changes between cells with no visible cell that produces the final
#      version used in Cells 22-25.
#   2. Deficiency threshold mismatch: notebook used deficit_pct > 30%,
#      but deficiency.py (and therefore the live main.py backend) uses > 20%.
#      This script uses deficiency.py's calculate_deficiency_percentage()
#      directly, so ML labels are IDENTICAL to what the live app calls
#      "Deficient" -- no second, drifting definition.
#   3. Hardcoded, duplicated nutrient content table (VITAMIN_C_FOODS etc.)
#      instead of the ontology. This script loads FOOD_NUTRIENTS/RDA_VALUES
#      from ontology_loader.py (i.e. from the actual .ttl) -- single source
#      of truth.
#   4. Symptom-only features (age/weight/symptoms) barely beat, or lost to,
#      the majority-class baseline (verified: 60.9% CV accuracy vs 67.3%
#      baseline for VitaminC) -- no nutrient's deficiency label showed a
#      significant chi-square association with any single symptom at n=110.
#      This script instead uses the FULL survey feature set (food intake +
#      symptoms + demographics + lifestyle), which is realistic (that's what
#      the live app already collects) and carries real signal (verified:
#      80-89% CV accuracy across the 4 viable nutrients, all comfortably
#      beating their majority baselines -- see printed report).
#   5. RandomForestClassifier's SHAP TreeExplainer returns a 3D array
#      (n_samples, n_features, n_classes) for binary classification in the
#      installed shap version -- the notebook only handled this for XGBoost
#      and crashed on RandomForest. Handled uniformly here.
#
# ALSO replaces the follow-up ML_Convergent.ipynb, which re-implemented the
# whole pipeline from scratch (its own ontology parsing, food parsing,
# symptom parsing) instead of reusing data_importer.py/deficiency.py, and in
# doing so reintroduced two of the same bugs plus a new one:
#   6. Symptom parsing bug reappeared verbatim (same substring-matching
#      issue as Food_Nutrition_deficit.ipynb) -- 8 of 11 symptoms never
#      match the raw checkbox text. Verified again on the actual CSV.
#   7. Portion size dropped entirely (frequency only) -- verified this
#      flips the Deficient/Adequate label for 9-20 of 110 respondents per
#      nutrient (8-18%), i.e. real label noise, not a cosmetic issue.
#   8. StandardScaler.fit_transform() was called on the full feature matrix
#      before the CV split (age/weight columns) -- leakage. Low practical
#      impact specifically for RandomForest (tree splits are scale-invariant
#      to monotonic transforms), but still bad practice and would matter for
#      a scale-sensitive model.
#   9. The notebook's "Convergent Validation" (Pearson r between rule-based
#      risk % and ML predicted probability) has a circularity problem: the
#      ML label IS the rule-based risk thresholded at >20%, so the "second
#      measurement" isn't independent -- this isn't convergent validity in
#      the psychometric sense (two independent instruments agreeing), it's
#      checking whether the model's probability output tracks the continuous
#      version of the very function it was trained to approximate. Still a
#      useful check (confirms the model learned the risk *gradient*, not
#      just a crude threshold) -- just relabeled and reframed correctly
#      below as "calibration / rank-agreement", with this caveat stated
#      explicitly rather than presented as independent validation.
#
# IMPORTANT LIMITATION (verified, not a bug in this script): of the 8
# nutrients with RDA data in the ontology, 4 -- Iron, Zinc, VitaminD,
# Calcium -- show 100% "Deficient" across all 110 respondents. This is a
# known ontology sparse-coverage issue (Zinc has values for only 3/27 foods,
# Calcium 5/27, VitaminD 4/27, Iron 12/27), not biological reality, and it
# means these 4 nutrients have ZERO class variance -- binary classification
# is not meaningful for them until more foods get nutrient data filled in
# (see the ontology_fill_template.csv effort already in progress). This
# script only trains models for the 4 nutrients that currently have real
# class variance: VitaminC, VitaminA, Carbohydrate, Protein.

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr

import data_importer as di
from deficiency import calculate_deficiency_percentage
from ontology_loader import RDA_VALUES, FOOD_NUTRIENTS

SURVEY_CSV = "Nutritional_Deficiency_Detection_Survey__Responses__-_Form_responses_1.csv"

SYMPTOM_LIST = [
    "Fatigue", "HairLoss", "GumBleeding", "Periodontitis", "NightBlindness",
    "DryFlakySkin", "BoneJointPain", "FrequentIllness", "PoorConcentration",
    "PaleConjunctiva", "Koilonychia",
]

# Property -> nutrient name, used only to report ontology coverage below
# (mirrors PROPERTY_TO_NUTRIENT in deficiency.py).
PROPERTY_TO_NUTRIENT = {
    "hasProteinContent_g_per_100g": "Protein", "hasIronContent_mg_per_100g": "Iron",
    "hasVitaminC_mg_per_100g": "VitaminC", "hasVitaminD_mcg_per_100g": "VitaminD",
    "hasVitaminA_mcg_per_100g": "VitaminA", "hasCalcium_mg_per_100g": "Calcium",
    "hasZincContent_mg_per_100g": "Zinc", "hasCarbohydrate_g_per_100g": "Carbohydrate",
}


def build_dataset(csv_path: str) -> pd.DataFrame:
    """
    Import the raw survey CSV via data_importer.py, then compute per-nutrient
    deficiency labels via deficiency.py's calculate_deficiency_percentage()
    -- the SAME function main.py calls for the live app -- so ML labels can
    never drift from what the app shows a user.
    """
    records = di.import_survey_csv(csv_path)
    nutrients = list(RDA_VALUES.keys())

    rows = []
    for r in records:
        # Matches the existing fallback in deficiency.py's RDA lookup
        # (rda_data.get(gender) or rda_data.get("male")): 'other' falls
        # back to male RDA. Flagging this rather than silently relying on
        # it -- if you want 'other' handled differently (e.g. averaged
        # male/female RDA, matching the approach used elsewhere in the
        # pipeline for gender), that needs a change in deficiency.py itself.
        gender_for_rda = r["gender"] if r["gender"] in ("male", "female") else "male"

        def_results = calculate_deficiency_percentage(
            r["food_intake_days_per_week"], gender=gender_for_rda, reported_symptoms=r["symptoms"]
        )

        row = {
            "row_index": r["row_index"], "age": r["age"], "gender": r["gender"],
            "religion": r["religion"], "height_cm": r["height_cm"], "weight_kg": r["weight_kg"],
            "sleep_category": r["sleep_category"],
            "screen_before_sleep_category": r["screen_before_sleep_category"],
            "sun_exposure_category": r["sun_exposure_category"],
            "living_arrangement": r["living_arrangement"],
        }
        for food, days in r["food_intake_days_per_week"].items():
            row[f"food__{food}"] = days
        for sym in SYMPTOM_LIST:
            row[f"symptom__{sym}"] = 1 if sym in r["symptoms"] else 0
        for nut in nutrients:
            row[f"label__{nut}"] = 1 if def_results[nut]["status"] == "Deficient" else 0
            row[f"risk__{nut}"] = def_results[nut]["deficit_percentage"]  # continuous 0-100
        rows.append(row)

    return pd.DataFrame(rows)


def report_ontology_coverage():
    from collections import Counter
    coverage = Counter()
    for _, props in FOOD_NUTRIENTS.items():
        for prop in props:
            if prop in PROPERTY_TO_NUTRIENT:
                coverage[PROPERTY_TO_NUTRIENT[prop]] += 1
    print(f"Ontology nutrient coverage ({len(FOOD_NUTRIENTS)} total foods):")
    for nut, cnt in coverage.most_common():
        print(f"  {nut:15s} {cnt}/{len(FOOD_NUTRIENTS)} foods have a value")
    print()


def get_feature_matrix(df: pd.DataFrame):
    cat_cols = ["gender", "religion", "sleep_category", "screen_before_sleep_category",
                "sun_exposure_category", "living_arrangement"]
    df = df.copy()
    # Explicit "Unknown" category rather than silently imputing -- e.g.
    # living_arrangement is missing for 47/110 respondents (real survey gap,
    # not a bug), so dropping or mean-imputing those rows/values would
    # misrepresent the data.
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    df_encoded = pd.get_dummies(df, columns=cat_cols)

    food_cols = [c for c in df.columns if c.startswith("food__")]
    symptom_cols = [c for c in df.columns if c.startswith("symptom__")]
    demo_cols = ["age", "height_cm", "weight_kg"]
    cat_dummy_cols = [c for c in df_encoded.columns if any(c.startswith(cc + "_") for cc in cat_cols)]

    feature_cols = food_cols + symptom_cols + demo_cols + cat_dummy_cols
    X = df_encoded[feature_cols].fillna(0)
    return X, feature_cols, df_encoded


def evaluate_nutrient(X, y, nutrient_name, n_splits=5, random_state=42):
    """5-fold stratified CV (not a single 70/30 split -- at n=110, a single
    split is noisy; CV gives a more defensible estimate) plus explicit
    majority-baseline comparison, which the notebook never reported."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=3,
        class_weight="balanced", random_state=random_state,
    )
    acc = cross_val_score(rf, X, y, cv=skf, scoring="accuracy")
    f1 = cross_val_score(rf, X, y, cv=skf, scoring="f1")
    proba = cross_val_predict(rf, X, y, cv=skf, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba)
    baseline = max(y.mean(), 1 - y.mean())

    return {
        "Nutrient": nutrient_name,
        "Deficient_prevalence": f"{y.mean():.1%}",
        "Majority_baseline_acc": f"{baseline:.1%}",
        "CV_accuracy": f"{acc.mean():.1%} (+/-{acc.std():.1%})",
        "CV_F1": f"{f1.mean():.3f}",
        "ROC_AUC": f"{auc:.3f}",
        "Beats_baseline": "YES" if acc.mean() > baseline else "NO",
    }


def shap_explain(X, y, feature_cols, nutrient_name, out_path, random_state=42):
    """
    Trains on a held-out split (SHAP needs a fixed fitted model, unlike the
    CV metrics above) and produces a SHAP summary plot.

    NOTE: TreeExplainer on RandomForestClassifier returns a 3D array
    (n_samples, n_features, n_classes) for binary classification in the
    installed shap version. The notebook only sliced this for XGBoost and
    crashed doing the same operation for RandomForest -- handled uniformly
    here for whichever shape comes back.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=3,
        class_weight="balanced", random_state=random_state,
    )
    rf.fit(X_train, y_train)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    shap_arr = np.array(shap_values)
    if shap_arr.ndim == 3:
        shap_vals_deficient = shap_arr[:, :, 1]  # class 1 = "Deficient"
    else:
        shap_vals_deficient = shap_arr

    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_vals_deficient, X_test, feature_names=feature_cols, show=False)
    plt.title(f"SHAP Feature Importance -- {nutrient_name} Deficiency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    importance = pd.DataFrame({
        "Feature": feature_cols,
        "Mean_abs_SHAP": np.abs(shap_vals_deficient).mean(axis=0),
    }).sort_values("Mean_abs_SHAP", ascending=False)
    return importance


def calibration_analysis(X, y_binary, rule_risk, nutrient_name, random_state=42):
    """
    Checks whether the ML model's cross-validated predicted probability
    tracks the ontology's continuous deficit_percentage (rule_risk, 0-100)
    for the SAME nutrient.

    ⚠️ NOT independent "convergent validity": y_binary is deficit_pct > 20%,
    so the model was trained to approximate a thresholded version of
    rule_risk itself. A strong correlation here mainly confirms the model
    learned the underlying risk gradient rather than a crude cutoff -- it is
    not evidence from a second, independent measurement instrument. Report
    it as calibration / rank-agreement, not as convergent validity.

    Uses cross_val_predict (out-of-fold predictions, same as the CV metrics
    in evaluate_nutrient) rather than in-sample .predict_proba() after
    .fit() on the same data, which would inflate the correlation.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=3,
        class_weight="balanced", random_state=random_state,
    )
    ml_proba = cross_val_predict(rf, X, y_binary, cv=skf, method="predict_proba")[:, 1]
    ml_risk = ml_proba * 100

    pearson_r, pearson_p = pearsonr(rule_risk, ml_risk)
    spearman_r, spearman_p = spearmanr(rule_risk, ml_risk)

    return {
        "Nutrient": nutrient_name,
        "Rule_risk_mean": round(float(np.mean(rule_risk)), 1),
        "ML_risk_mean": round(float(np.mean(ml_risk)), 1),
        "Pearson_r": round(pearson_r, 3),
        "Pearson_p": round(pearson_p, 4),
        "Spearman_rho": round(spearman_r, 3),
        "Spearman_p": round(spearman_p, 4),
    }, ml_risk


def plot_calibration_grid(rule_risks: dict, ml_risks: dict, out_path: str):
    n = len(rule_risks)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, nut in zip(axes, rule_risks.keys()):
        rr, mr = rule_risks[nut], ml_risks[nut]
        pearson_r, _ = pearsonr(rr, mr)
        spearman_r, _ = spearmanr(rr, mr)
        ax.scatter(rr, mr, alpha=0.6, s=40, c="steelblue")
        ax.plot([0, 100], [0, 100], "r--", linewidth=1.5, label="y = x")
        ax.set_xlabel("Ontology risk % (rule-based)")
        ax.set_ylabel("ML predicted risk % (CV)")
        ax.set_title(f"{nut}\nPearson r={pearson_r:.2f}, Spearman ρ={spearman_r:.2f}")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_deployment_bundle(X, feature_cols, df_encoded, viable_nutrients, shap_importances,
                            out_path="ml_deployment_bundle.joblib", random_state=42):
    """
    Trains a FINAL model per viable nutrient on the FULL dataset (all 110
    respondents, no holdout) -- appropriate here because CV in Step 3 already
    validated performance; a final deployed model should use all available
    data (same practice noted in the ML_Convergent.ipynb comments, which was
    correct on this specific point).

    Saves everything main.py needs to serve predictions WITHOUT recomputing
    SHAP per request:
      - 'feature_cols': exact ordered column list -- a live request must be
        encoded into this exact same schema (same one-hot columns, same
        order) or the model's predictions are meaningless.
      - 'models': one fitted RandomForestClassifier per viable nutrient.
      - 'importance': the top-5 SHAP features per nutrient computed offline
        in Step 4 -- served as a STATIC "what generally drives this"
        explanation rather than live per-request SHAP (cheaper, and avoids
        shipping the shap library + background data into the request path).
    """
    models = {}
    for nut in viable_nutrients:
        y = df_encoded[f"label__{nut}"]
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=3,
            class_weight="balanced", random_state=random_state,
        )
        rf.fit(X, y)
        models[nut] = rf

    bundle = {
        "feature_cols": feature_cols,
        "models": models,
        "importance": shap_importances,  # {nutrient: [(feature, mean_abs_shap), ...]}
    }
    joblib.dump(bundle, out_path)
    print(f"Saved deployment bundle to {out_path} ({len(viable_nutrients)} nutrients, {len(feature_cols)} features)")


def main():
    print("=" * 70)
    print("STEP 1: Load survey + compute ontology-driven deficiency labels")
    print("=" * 70)
    df = build_dataset(SURVEY_CSV)
    print(f"Loaded {len(df)} respondents\n")

    report_ontology_coverage()

    nutrients = list(RDA_VALUES.keys())
    print("Class balance per nutrient:")
    viable_nutrients = []
    for nut in nutrients:
        y = df[f"label__{nut}"]
        prevalence = y.mean()
        has_variance = y.nunique() > 1
        flag = "" if has_variance else "  <-- NO CLASS VARIANCE, cannot train (see coverage above)"
        print(f"  {nut:15s} deficient={prevalence:.1%}{flag}")
        if has_variance:
            viable_nutrients.append(nut)
    print(f"\nViable nutrients for ML (have both classes): {viable_nutrients}\n")

    print("=" * 70)
    print("STEP 2: Build full-survey feature matrix (food + symptom + demo + lifestyle)")
    print("=" * 70)
    X, feature_cols, df_encoded = get_feature_matrix(df)
    print(f"Total features: {len(feature_cols)}\n")

    print("=" * 70)
    print("STEP 3: Train + 5-fold CV evaluate each viable nutrient")
    print("=" * 70)
    results = []
    for nut in viable_nutrients:
        y = df_encoded[f"label__{nut}"]
        results.append(evaluate_nutrient(X, y, nut))
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    results_df.to_csv("model_results.csv", index=False)
    print("\nSaved model_results.csv")

    print("\n" + "=" * 70)
    print("STEP 4: SHAP explainability for each viable nutrient")
    print("=" * 70)
    shap_importances = {}
    for nut in viable_nutrients:
        y = df_encoded[f"label__{nut}"]
        out_path = f"shap_{nut}.png"
        importance = shap_explain(X, y, feature_cols, nut, out_path)
        print(f"\nTop 5 features for {nut} deficiency:")
        print(importance.head(5).to_string(index=False))
        print(f"Saved {out_path}")
        shap_importances[nut] = list(importance.head(5).itertuples(index=False, name=None))

    print("\n" + "=" * 70)
    print("STEP 5: Calibration / rank-agreement with the ontology's continuous risk score")
    print("(NOT independent convergent validity -- see calibration_analysis() docstring)")
    print("=" * 70)
    calib_results = []
    rule_risks, ml_risks = {}, {}
    for nut in viable_nutrients:
        y = df_encoded[f"label__{nut}"]
        rule_risk = df_encoded[f"risk__{nut}"].values
        calib, ml_risk = calibration_analysis(X, y, rule_risk, nut)
        calib_results.append(calib)
        rule_risks[nut] = rule_risk
        ml_risks[nut] = ml_risk
        sig = "significant" if calib["Spearman_p"] < 0.05 else "NOT significant"
        print(f"  {nut:15s} Pearson r={calib['Pearson_r']:.3f} (p={calib['Pearson_p']:.4f}), "
              f"Spearman ρ={calib['Spearman_rho']:.3f} (p={calib['Spearman_p']:.4f}) -- {sig}")

    calib_df = pd.DataFrame(calib_results)
    calib_df.to_csv("calibration_results.csv", index=False)
    plot_calibration_grid(rule_risks, ml_risks, "calibration_grid.png")
    print("\nSaved calibration_results.csv and calibration_grid.png")

    print("\n" + "=" * 70)
    print("STEP 6: Save deployment bundle for main.py's /ml-insight endpoint")
    print("=" * 70)
    save_deployment_bundle(X, feature_cols, df_encoded, viable_nutrients, shap_importances)


if __name__ == "__main__":
    main()
