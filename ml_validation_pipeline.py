"""
ml_validation_pipeline.py
==========================
Complete ML + Validation Pipeline with Dummy Data (CORRECTED VERSION)

Fixes applied to the original draft:
  1. SHAP output-shape bug: newer SHAP versions return a 3D array for
     TreeExplainer binary classification, not a list of two 2D arrays.
     The old code crashed on this. Now handles both formats.
  2. Iron's dummy symptoms now include PoorConcentration (the real ontology
     lists Iron -> Fatigue, HairLoss, PoorConcentration, PaleConjunctiva,
     Koilonychia -- the original draft left PoorConcentration as pure
     random noise, disconnected from Iron).
  3. VitaminA's dummy symptoms now use ONLY NightBlindness (the real
     ontology does not link VitaminA to DryFlakySkin -- that symptom
     belongs to VitaminD only. The original draft incorrectly shared it
     between both nutrients).
  4. NOTE ON Protein: your ontology currently has NO causesSymptom link
     for Protein at all. The Protein->Fatigue/HairLoss link below is
     DeepSeek's own assumption, not grounded in your ontology. Either
     add a real causesSymptom link for Protein in the .ttl, or drop
     Protein from the symptom-based ML classification and treat it
     separately (e.g. from FFQ intake alone) -- your call, flagged here
     so it doesn't silently slip into your results section unexamined.

Run this file directly to see the full pipeline output for all 7 nutrients.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shap
from scipy.stats import pearsonr, spearmanr, ttest_ind, f_oneway, chi2_contingency
import warnings
warnings.filterwarnings('ignore')


# ----------------------------
# 1. DUMMY DATA GENERATOR
# ----------------------------

def generate_dummy_data(n=200):
    """
    Generates realistic dummy data with known patterns, aligned to the
    actual ontology's causesSymptom links (see fixes #2 and #3 above).
    """
    np.random.seed(42)

    # ---- Iron: Fatigue, HairLoss, PaleConjunctiva, Koilonychia, PoorConcentration ----
    iron_prob = np.random.uniform(0.2, 0.8, n)
    fatigue = (iron_prob + np.random.normal(0, 0.1, n) > 0.5).astype(int)
    hairloss = (iron_prob * 0.8 + np.random.normal(0, 0.1, n) > 0.5).astype(int)
    pale_conj = (iron_prob * 0.7 + np.random.normal(0, 0.1, n) > 0.5).astype(int)
    koilonychia = (iron_prob * 0.6 + np.random.normal(0, 0.1, n) > 0.5).astype(int)
    poor_concentration = (iron_prob * 0.5 + np.random.normal(0, 0.1, n) > 0.5).astype(int)

    # ---- VitaminC: GumBleeding, Periodontitis, FrequentIllness ----
    vitc_prob = np.random.uniform(0.2, 0.8, n)
    gum_bleed = (vitc_prob * 0.8 + np.random.normal(0, 0.1, n) > 0.5).astype(int)
    periodontitis = (vitc_prob * 0.7 + np.random.normal(0, 0.1, n) > 0.5).astype(int)
    freq_illness = (vitc_prob * 0.6 + np.random.normal(0, 0.1, n) > 0.5).astype(int)

    # ---- VitaminD: BoneJointPain, DryFlakySkin ----
    vitd_prob = np.random.uniform(0.2, 0.8, n)
    bone_pain = (vitd_prob * 0.8 + np.random.normal(0, 0.1, n) > 0.5).astype(int)
    dry_skin = (vitd_prob * 0.7 + np.random.normal(0, 0.1, n) > 0.5).astype(int)

    # ---- VitaminA: NightBlindness ONLY (fixed -- was incorrectly sharing DryFlakySkin) ----
    vita_prob = np.random.uniform(0.2, 0.8, n)
    night_blind = (vita_prob * 0.8 + np.random.normal(0, 0.1, n) > 0.5).astype(int)

    zinc_prob = np.random.uniform(0.2, 0.8, n)
    ca_prob = np.random.uniform(0.2, 0.8, n)
    protein_prob = np.random.uniform(0.2, 0.8, n)  # see NOTE above -- not ontology-grounded

    # ---- Lifestyle ----
    sleep = np.random.choice(['4-6', '6-8', '8+', '<4'], size=n, p=[0.3, 0.4, 0.2, 0.1])
    screen = np.random.choice(['<30min', '30-60min', '1-2hr', '2hr+'], size=n, p=[0.2, 0.3, 0.3, 0.2])
    bmi = np.random.normal(22, 4, n).clip(15, 40)

    # ---- Ground Truth Deficiency Labels ----
    iron_def = ((iron_prob > 0.5) & ((fatigue + hairloss + pale_conj + koilonychia + poor_concentration) >= 3)).astype(int)
    vitc_def = ((vitc_prob > 0.5) & ((gum_bleed + periodontitis + freq_illness) >= 2)).astype(int)
    vitd_def = ((vitd_prob > 0.5) & ((bone_pain + dry_skin) >= 1)).astype(int)
    vita_def = ((vita_prob > 0.5) & (night_blind == 1)).astype(int)
    ca_def = ((ca_prob > 0.5) & (bone_pain == 1)).astype(int)
    zinc_def = ((zinc_prob > 0.5) & ((hairloss + freq_illness) >= 1)).astype(int)
    protein_def = ((protein_prob > 0.5) & ((fatigue + hairloss) >= 1)).astype(int)

    # ---- Rule-based Risk % (for Convergent Validity) ----
    iron_risk = np.clip((fatigue + hairloss + pale_conj + koilonychia + poor_concentration) / 5 * 100 + np.random.normal(0, 5, n), 0, 100)
    vitc_risk = np.clip((gum_bleed + periodontitis + freq_illness) / 3 * 100 + np.random.normal(0, 5, n), 0, 100)
    vitd_risk = np.clip((bone_pain + dry_skin) / 2 * 100 + np.random.normal(0, 5, n), 0, 100)
    vita_risk = np.clip(night_blind * 100 + np.random.normal(0, 5, n), 0, 100)
    ca_risk = np.clip(bone_pain * 100 + np.random.normal(0, 5, n), 0, 100)
    zinc_risk = np.clip((hairloss + freq_illness) / 2 * 100 + np.random.normal(0, 5, n), 0, 100)
    protein_risk = np.clip((fatigue + hairloss) / 2 * 100 + np.random.normal(0, 5, n), 0, 100)

    # ---- Known-Groups (for validation) ----
    living = np.random.choice(['Home', 'Mess/Hostel', 'Shared'], size=n, p=[0.4, 0.4, 0.2])
    living_shift = np.where(living == 'Mess/Hostel', 1.2, 1.0)
    diet = np.random.choice(['Non-veg', 'Vegetarian'], size=n, p=[0.7, 0.3])
    diet_shift = np.where(diet == 'Vegetarian', 1.3, 1.0)

    iron_risk = np.clip(iron_risk * living_shift * diet_shift, 0, 100)
    zinc_risk = np.clip(zinc_risk * living_shift * diet_shift, 0, 100)

    df = pd.DataFrame({
        'Fatigue': fatigue, 'HairLoss': hairloss, 'GumBleeding': gum_bleed,
        'Periodontitis': periodontitis, 'NightBlindness': night_blind,
        'DryFlakySkin': dry_skin, 'BoneJointPain': bone_pain,
        'FrequentIllness': freq_illness,
        'PoorConcentration': poor_concentration,   # fixed: tied to Iron, not random
        'PaleConjunctiva': pale_conj, 'Koilonychia': koilonychia,
        'SleepHours': sleep, 'ScreenTime': screen, 'BMI': bmi,
        'IronDef': iron_def, 'ProteinDef': protein_def, 'VitaminCDef': vitc_def,
        'VitaminDDef': vitd_def, 'VitaminADef': vita_def, 'CalciumDef': ca_def,
        'ZincDef': zinc_def,
        'IronRisk': iron_risk, 'ProteinRisk': protein_risk, 'VitaminCRisk': vitc_risk,
        'VitaminDRisk': vitd_risk, 'VitaminARisk': vita_risk, 'CalciumRisk': ca_risk,
        'ZincRisk': zinc_risk,
        'Living': living, 'Diet': diet
    })
    return df


# ----------------------------
# 2. ML PIPELINE
# ----------------------------

def run_ml_pipeline(df, nutrient):
    """Train Logistic Regression and Random Forest for a given nutrient."""
    feature_cols = ['Fatigue', 'HairLoss', 'GumBleeding', 'Periodontitis',
                    'NightBlindness', 'DryFlakySkin', 'BoneJointPain',
                    'FrequentIllness', 'PoorConcentration', 'PaleConjunctiva',
                    'Koilonychia', 'SleepHours', 'ScreenTime', 'BMI']

    X = df[feature_cols].copy()
    X = pd.get_dummies(X, columns=['SleepHours', 'ScreenTime'], drop_first=True)
    y = df[f'{nutrient}Def']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    metrics = {
        'LR': {
            'Accuracy': accuracy_score(y_test, lr_pred),
            'Precision': precision_score(y_test, lr_pred, zero_division=0),
            'Recall': recall_score(y_test, lr_pred, zero_division=0),
            'F1': f1_score(y_test, lr_pred, zero_division=0),
        },
        'RF': {
            'Accuracy': accuracy_score(y_test, rf_pred),
            'Precision': precision_score(y_test, rf_pred, zero_division=0),
            'Recall': recall_score(y_test, rf_pred, zero_division=0),
            'F1': f1_score(y_test, rf_pred, zero_division=0),
        },
    }
    metrics['LR']['CV_Mean'] = cross_val_score(lr, X_train_scaled, y_train, cv=5).mean()
    metrics['RF']['CV_Mean'] = cross_val_score(rf, X_train, y_train, cv=5).mean()

    # ---- SHAP (FIXED: handles both old-list and new-3D-array output shapes) ----
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]                 # older SHAP: list[class0, class1]
    elif np.array(shap_values).ndim == 3:
        shap_vals = shap_values[:, :, 1]            # newer SHAP: (samples, features, classes)
    else:
        shap_vals = shap_values

    feature_names = X.columns.tolist()
    shap_summary = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Mean': np.abs(shap_vals).mean(axis=0)
    }).sort_values('SHAP_Mean', ascending=False)

    return {
        'metrics': metrics,
        'rf_proba': rf_proba,
        'lr_proba': lr_proba,
        'shap_summary': shap_summary,
        'test_index': X_test.index,   # NEW: needed so convergent_validity compares the SAME rows
    }


# ----------------------------
# 3. VALIDATION FUNCTIONS
# ----------------------------

def convergent_validity(df, nutrient, ml_proba, test_index):
    """
    Pearson/Spearman correlation between rule-based risk % and ML probability.
    IMPORTANT: ml_proba only has values for the TEST SET rows (after the
    train/test split), so rule_risk must be filtered to the same rows --
    otherwise the arrays have mismatched lengths (this crashed in the
    original draft: 200 rows vs 60 test-set predictions).
    """
    rule_risk = df.loc[test_index, f'{nutrient}Risk']
    pearson_r, pearson_p = pearsonr(rule_risk, ml_proba)
    spearman_r, spearman_p = spearmanr(rule_risk, ml_proba)
    return {'Pearson_r': pearson_r, 'Pearson_p': pearson_p,
            'Spearman_r': spearman_r, 'Spearman_p': spearman_p}


def known_groups_validity(df, nutrient):
    """t-test (Diet, 2 groups) + ANOVA (Living, 3 groups) + Chi-square."""
    results = {}
    living_groups = [df[df['Living'] == g][f'{nutrient}Risk'] for g in df['Living'].unique()]
    if len(living_groups) >= 2:
        f_stat, p_val = f_oneway(*living_groups)
        results['Living_ANOVA'] = {'F': f_stat, 'p': p_val}

    diet_groups = [df[df['Diet'] == g][f'{nutrient}Risk'] for g in df['Diet'].unique()]
    if len(diet_groups) == 2:
        t_stat, p_val = ttest_ind(diet_groups[0], diet_groups[1])
        results['Diet_t-test'] = {'t': t_stat, 'p': p_val}

    contingency = pd.crosstab(df['Diet'], df[f'{nutrient}Def'])
    if contingency.shape[1] > 1:
        chi2, p, dof, expected = chi2_contingency(contingency)
        results['Diet_Chi2'] = {'Chi2': chi2, 'p': p}

    return results


# ----------------------------
# 4. SAMPLE CASES (for Content Validity -- expert review)
# ----------------------------

def generate_sample_cases(df, n_cases=10):
    sample_df = df.sample(n_cases, random_state=42)
    cases = []
    for idx, row in sample_df.iterrows():
        cases.append({
            'ID': idx,
            'Symptoms': {k: int(row[k]) for k in
                ['Fatigue', 'HairLoss', 'GumBleeding', 'Periodontitis', 'NightBlindness',
                 'DryFlakySkin', 'BoneJointPain', 'FrequentIllness', 'PoorConcentration',
                 'PaleConjunctiva', 'Koilonychia']},
            'Lifestyle': {'Sleep': row['SleepHours'], 'ScreenTime': row['ScreenTime'],
                          'BMI': round(row['BMI'], 1)},
            'Rule_Risks': {n: round(row[f'{n}Risk'], 1) for n in
                ['Iron', 'Protein', 'VitaminC', 'VitaminD', 'VitaminA', 'Calcium', 'Zinc']},
            'Deficiency_Labels': {n: int(row[f'{n}Def']) for n in
                ['Iron', 'Protein', 'VitaminC', 'VitaminD', 'VitaminA', 'Calcium', 'Zinc']},
        })
    return cases


# ----------------------------
# 5. MAIN EXECUTION
# ----------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("NUTRITION DEFICIENCY ML + VALIDATION PIPELINE (dummy data)")
    print("=" * 70)

    df = generate_dummy_data(200)
    print(f"\nData shape: {df.shape}")

    nutrients = ['Iron', 'Protein', 'VitaminC', 'VitaminD', 'VitaminA', 'Calcium', 'Zinc']

    print(f"\n{'Nutrient':<12} {'LR Acc':<8} {'RF Acc':<8} {'LR F1':<8} {'RF F1':<8} {'Pearson r':<10}")
    print("-" * 60)
    for nutrient in nutrients:
        result = run_ml_pipeline(df, nutrient)
        conv = convergent_validity(df, nutrient, result['rf_proba'], result['test_index'])
        m = result['metrics']
        print(f"{nutrient:<12} {m['LR']['Accuracy']:.3f}    {m['RF']['Accuracy']:.3f}    "
              f"{m['LR']['F1']:.3f}    {m['RF']['F1']:.3f}    {conv['Pearson_r']:.3f}")

    print("\nGenerating 10 sample cases for expert (Professor) content-validity review...")
    cases = generate_sample_cases(df, 10)
    print(f"Done -- {len(cases)} cases generated. See sample_cases_for_review.md if exported.")
