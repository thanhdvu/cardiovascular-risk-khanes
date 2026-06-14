# KNHANES 고혈압 위험 예측 프로젝트

## 1. 프로젝트 개요

이 프로젝트는 KNHANES 2017-2023 데이터를 사용하여 한국 젊은 성인(19-39세)의 고혈압 위험을 예측하기 위한 수업용 머신러닝 프로젝트이다.

교수님 피드백을 반영하여 고혈압 target은 ACC/AHA 2017 기준인 `SBP >= 130` 또는 `DBP >= 80`으로 정의하였다. `covid_period`는 COVID-19의 인과효과가 아니라 조사 시기와 측정환경 변화가 함께 반영된 변수로 제한해서 해석하였다.

## 2. 데이터셋

- 자료: KNHANES 2017-2023
- 대상: 19-39세 젊은 성인
- 최종 분석 대상자 수: 10,319명
- 고혈압 그룹: 2,372명
- 고혈압 비율: 23.0%

HDL은 측정 시기별 변화 가능성이 있어 주분석 모델에서 제외하였다. LDL은 전체 결측률이 약 68%로 높아 주분석에서 제외하였다. `HE_sbp`, `HE_dbp`는 target 생성에만 사용하고 모델 feature에서는 제외하였다.

## 3. Notebook 실행 순서

1. `01_data_understanding.ipynb`
2. `02_data_cleaning_eda.ipynb`
3. `05_measurement_sensitivity.ipynb`
4. `03_pycaret_modeling.ipynb`

## 4. 모델 요약

PyCaret workflow를 중심으로 `setup()` → `compare_models()` → 상위 모델 선택 → `tune_model()` → `plot_model()` → SHAP 해석 순서로 수행하였다. Class imbalance는 130/80 기준 적용으로 완화하고, PyCaret의 SMOTE 설정과 최종 Logistic Regression의 `class_weight='balanced'`를 함께 사용하였다. Logistic Regression, Random Forest, LightGBM, XGBoost, CatBoost를 비교하였고, 최종 모델은 AUC와 recall을 고려하여 Logistic Regression으로 선정하였다.

Holdout set 기준 최종 모델 결과:

- Precision: 0.406
- Recall: 0.660
- F1: 0.503
- ROC AUC: 0.765
- Average Precision: 0.511

## 5. Streamlit 사용

아래 명령어로 예측 데모 앱을 실행할 수 있다.

```bash
streamlit run app.py
```

앱에는 예측 입력 화면과 사용자 피드백 저장 기능이 포함되어 있다. 피드백은 `processed_data/user_feedback.csv`에 저장된다.

