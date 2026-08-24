# run_ml_on_real_data.py
#
# Runs run_ml_pipeline() + convergent_validity() on the real survey data,
# after build_ml_dataframe() has flattened it into the shape the ML
# pipeline expects.

from build_ml_dataframe import build_ml_dataframe, NUTRIENTS
from ml_validation_pipeline import run_ml_pipeline, convergent_validity, known_groups_validity

df = build_ml_dataframe("survey_data.csv")

# run_ml_pipeline() can't handle NaN features -- drop rows missing lifestyle data
before = len(df)
df_model = df.dropna(subset=["BMI", "SleepHours", "ScreenTime"])
print(f"Usable rows for ML: {len(df_model)} / {before} "
      f"(dropped {before - len(df_model)} with missing BMI/SleepHours/ScreenTime)")

if df["Living"].isna().sum() > 0:
    print(f"NOTE: 'Living' has {df['Living'].isna().sum()} missing values -- "
          f"known_groups_validity()'s ANOVA on Living will drop those rows too.")

print()
MIN_CLASS_COUNT = 5  # need at least this many in the minority class to stratify-split + get a meaningful test set

for nutrient in NUTRIENTS:
    y = df_model[f"{nutrient}Def"]
    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < MIN_CLASS_COUNT:
        print(f"{nutrient:10s}: skipped -- class counts {counts.to_dict()}, "
              f"minority class has fewer than {MIN_CLASS_COUNT} respondents "
              f"(can't reliably train/stratify a classifier on it -- this usually means "
              f"almost everyone is Deficient/Adequate for this nutrient, which is itself "
              f"worth reporting descriptively even without an ML model).")
        continue
    result = run_ml_pipeline(df_model, nutrient)
    conv = convergent_validity(df_model, nutrient, result["rf_proba"], result["test_index"])
    m = result["metrics"]
    print(f"{nutrient:10s}: LR Acc={m['LR']['Accuracy']:.3f}  RF Acc={m['RF']['Accuracy']:.3f}  "
          f"Pearson r={conv['Pearson_r']:.3f} (p={conv['Pearson_p']:.3f})")
