def get_bmi_category(bmi: float) -> dict:
    """
    WHO Asian BMI cutoffs (WHO Expert Consultation, 2004) - lower thresholds
    than the standard global WHO cutoffs (25/30), because South/East Asian
    populations show higher health risk at lower BMI values.
    NOTE: Begum et al. (2024, BDHS pregnant women study) used the STANDARD
    global cutoffs (18.5 / 25 / 30), not the Asian-specific ones. Pick one
    and cite it consistently -- don't mix cutoffs across your paper.
    """
    if bmi < 18.5:
        return {"category": "Underweight", "bangla": "অপুষ্ট/কম ওজন", "risk": "High"}
    elif bmi < 23.0:
        return {"category": "Normal", "bangla": "স্বাভাবিক", "risk": "Low"}
    elif bmi < 27.5:
        return {"category": "Overweight", "bangla": "অতিরিক্ত ওজন", "risk": "Moderate"}
    else:
        return {"category": "Obese", "bangla": "মোটা/স্থুলকায়", "risk": "High"}

if __name__ == "__main__":
    for test_bmi in [17.0, 18.5, 20.0, 22.9, 23.0, 27.4, 27.5, 30.0]:
        print(f"BMI {test_bmi:5.1f} -> {get_bmi_category(test_bmi)}")