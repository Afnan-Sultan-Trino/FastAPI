# Ontology-Driven Explainable AI for Nutritional Deficiency Risk Detection & Local Food Recommendation

## 📖 Overview
This repository contains an end-to-end, explainable AI framework designed to detect nutritional deficiency risks among Bangladeshi university students. By combining a rule-based assessment (dietary intake vs. RDA), a Machine Learning model (symptom/lifestyle-based), Explainable AI (SHAP), and a custom OWL ontology, this system provides personalized, culturally relevant, and affordable local food recommendations. 

## 🚨 Problem Statement
Nutritional deficiencies are common among university students in Bangladesh due to poor dietary habits, irregular schedules, limited income, and a lack of accessible clinical screening services. Existing digital tools often isolate dietary data from symptom/lifestyle data, lack explainability, and fail to consider religious, cultural, and financial constraints when recommending foods.

## 💡 Our Solution
The framework bridges the gap between clinical nutrition and practical, everyday dietary choices by integrating:
1. **Bangladeshi Food Ontology:** Models the relationships between local foods, nutrients, and deficiency symptoms.
2. **Rule-Based Risk Assessment:** Calculates continuous deficiency risk using Food Frequency Questionnaire (FFQ) data, portion sizes, and gender-specific Recommended Dietary Allowances (RDAs).
3. **Machine Learning Prediction:** Uses Random Forest classifiers trained on symptom and lifestyle data to predict deficiency status.
4. **Explainable AI (XAI):** Utilizes SHAP (SHapley Additive exPlanations) to provide transparency into which symptoms and lifestyle factors are driving the ML predictions.
5. **Personalized Recommendation Engine:** Generates context-aware local food suggestions, complete with portion sizes, filtered by religion and budget constraints.

## ⚙️ Pipeline Workflow
1. **Data Collection:** Google Forms survey capturing demographics, lifestyle, FFQ (27 items), portion sizes, and 11 physical symptoms.
2. **Rule-Based Assessment:** Nutrient intake is calculated and compared against RDAs to derive a continuous deficiency risk score.
3. **ML Prediction:** Random Forest classifies deficiency status for viable nutrients based on symptoms and lifestyle.
4. **Explainability:** SHAP values identify the most influential features (e.g., specific foods or symptoms) for each prediction.
5. **Validation:** Pearson and Spearman correlations compare the rule-based risk and ML-predicted risk to assess calibration and rank agreement.
6. **Recommendation:** The system identifies nutrient gaps and suggests affordable, religion-compliant local Bangladeshi foods to fill them.

## 🛠️ Tech Stack
*   **Ontology & Reasoning:** OWL (Web Ontology Language), Protégé, Manchester-syntax DL Queries
*   **Machine Learning:** Random Forest (Scikit-learn)
*   **Explainable AI (XAI):** SHAP
*   **Data Analysis:** Pearson and Spearman correlation, 5-fold Stratified Cross-Validation
*   **Data Collection:** Google Forms

## 📊 Dataset & Targets
*   **Participants:** 102 valid responses from Bangladeshi university students (out of 110 collected).
*   **Data Features:** Demographics, lifestyle, 27-item FFQ, 11 physical symptoms.
*   **Viable ML Targets:** Out of 8 nutrients, only 5 were viable for binary classification due to class variance (Iron, Calcium, and Vitamin D had ~100% deficiency prevalence, making them non-viable).
    *   Zinc (93.1% deficient)
    *   Vitamin C (65.7% deficient)
    *   Vitamin A (61.8% deficient)
    *   Protein (54.9% deficient)
    *   Carbohydrate (52.0% deficient)

## 🧠 Model Performance
The Random Forest models achieved strong discriminative ability across all five modeled nutrients using 5-fold stratified cross-validation.
*   **Cross-Validation Accuracy:** 84.2% – 93.2%
*   **ROC-AUC:** 0.927 – 0.980
    *   *Highest Accuracy:* Zinc (93.2%)
    *   *Highest ROC-AUC:* Vitamin A (0.980)

**Top SHAP Features Identified:**
*   **Vitamin A:** LalShak, PalongShak
*   **Vitamin C:** Pepe, Aam
*   **Protein:** Murgir Mangsho (Chicken), Kamalalebu
*   **Carbohydrate:** Atta, Bhat
*   **Zinc:** Gorur Mangsho (Beef), Murgir Mangsho

## 📈 Validation
Rule-based and ML risk estimates were compared to assess calibration (since ML labels were derived from the 20% rule-based threshold). 
*   **High Rank Agreement (Spearman ρ > 0.80):** Vitamin A, Vitamin C, Protein, Carbohydrate.
*   **Moderate Agreement (ρ = 0.585):** Zinc (reflecting limited zinc-rich foods in the current ontology).

## ⚠️ Limitations & Future Work
*   **Limitations:** The study is limited to university students, the ontology currently covers a limited set of foods, and the system is a research prototype that has not yet undergone clinical/medical validation.
*   **Future Work:** Expand the Bangladeshi food ontology to include a wider variety of items, increase the sample size, conduct clinical validation, and explore deployment via mobile or wearable systems.

## 📚 References
1. Al-Nazer et al., "User's profile ontology-based semantic framework for personalized food and nutrition recommendation," *Procedia Computer Science*, 2014.
2. Begum et al., "Machine learning prediction of nutritional status among pregnant women in Bangladesh," *PLOS ONE*, 2024.
3. Institute of Nutrition and Food Science, *Food Composition Table for Bangladesh*, 2013.
4. Kabir et al., "Factors influencing eating behavior and dietary intake among resident students in a public university in Bangladesh," *PLOS ONE*, 2018.
5. Shill et al., "Prevalence of iron-deficiency anaemia among university students in Noakhali region, Bangladesh," *J Health Popul Nutr*, 2014.
6. USDA FoodData Central, 2019.
7. WHO, "Malnutrition," Fact sheet.
8. NIH, Office of Dietary Supplements (Iron, Vitamin A, Vitamin C, Zinc Fact Sheets).
