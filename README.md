# KNHANES 고혈압 위험 예측 프로젝트

## 1. 프로젝트 개요

이 프로젝트는 KNHANES 2017-2023 데이터를 사용하여 한국 젊은 성인(19-39세)의 고혈압 위험을 예측하는 수업용 머신러닝 프로젝트이다. 데이터 이해, 전처리, EDA, 모델 비교, 최종 모델 해석, 간단한 Streamlit 데모까지 하나의 흐름으로 진행하였다.

## 2. 데이터셋

데이터는 KNHANES 2017-2023 원자료를 바탕으로 정리하였다. 고혈압 target은 혈압 측정값으로 만들기 때문에, SBP 또는 DBP가 결측인 대상자는 target 생성 전에 제외하였다.

- 최종 분석 대상자 수: 10,319명
- 고혈압 그룹: 575명
- Target 정의: SBP >= 140 또는 DBP >= 90

정리된 데이터는 `processed_data/` 폴더에 저장되어 있다.

## 3. Notebook 실행 순서

Notebook은 아래 순서로 실행한다.

1. `01_data_understanding.ipynb`
2. `02_data_cleaning_eda.ipynb`
3. `03_pycaret_modeling.ipynb`

## 4. 모델 요약

AUC를 주요 기준으로 여러 classification model을 비교하였다. 최종 모델은 Logistic Regression으로 선택하였다. AUC가 가장 높았고, 수업 프로젝트 수준에서 설명하기 쉽고 해석하기 쉬운 모델이기 때문이다.

최종 모델 파일은 아래 경로에 저장되어 있다.

`models/final_hypertension_model.pkl`

## 5. Streamlit 실행 방법

필요한 패키지를 설치한 뒤 Streamlit 앱을 실행한다.

```bash
pip install -r requirements.txt
streamlit run app.py
```

이 앱은 수업 프로젝트용 데모이며 실제 의료 진단을 대체하지 않는다.
