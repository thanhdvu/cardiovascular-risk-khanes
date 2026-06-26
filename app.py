from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
from pycaret.classification import load_model as pycaret_load_model
from pycaret.classification import predict_model


MODEL_PATH = "models/final_hypertension_model"
FEEDBACK_PATH = Path("processed_data/user_feedback.csv")


st.set_page_config(
    page_title="KNHANES Hypertension Prediction",
    layout="wide",
)


@st.cache_resource
def get_model():
    return pycaret_load_model(MODEL_PATH)


model = get_model()


def medical_label(label, description):
    st.markdown(
        f"""
        <div class="field-label">
            {label}
            <span class="tooltip-dot">?</span>
            <span class="tooltip-text">{description}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


income_labels = {
    1.0: "하위 25% 소득 그룹",
    2.0: "25-50% 소득 그룹",
    3.0: "50-75% 소득 그룹",
    4.0: "상위 25% 소득 그룹",
}

education_labels = {
    2.0: "2.0 - 초졸 이하",
    3.0: "3.0 - 중졸",
    4.0: "4.0 - 고졸",
    5.0: "5.0 - 대학 재학",
    6.0: "6.0 - 전문대졸",
    7.0: "7.0 - 대졸",
    8.0: "8.0 - 대학원+",
}

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f6fbff 0%, #f8fffb 45%, #ffffff 100%);
    }
    .block-container {
        padding-top: 2.8rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }
    h1 {
        font-size: 2.25rem !important;
        margin-bottom: 0.2rem !important;
    }
    h2, h3, p, label, span, div {
        color: #17233c;
    }
    h2, h3 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.55rem !important;
    }
    div[data-testid="stWidgetLabel"] label p {
        color: #17233c !important;
        font-weight: 700 !important;
        font-size: 0.86rem !important;
        white-space: nowrap !important;
    }
    div[data-testid="stWidgetLabel"] {
        margin-bottom: 0.25rem;
    }
    div[data-testid="stTooltipHoverTarget"] svg,
    button[aria-label="help"] svg,
    svg[aria-label="help"] {
        color: #176b87 !important;
        fill: #176b87 !important;
        opacity: 1 !important;
        width: 1rem !important;
        height: 1rem !important;
    }
    div[data-testid="stMarkdownContainer"] p {
        color: #24364f;
    }
    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextArea"] textarea {
        background-color: #ffffff !important;
        color: #17233c !important;
        border: 1px solid #cbdff5 !important;
        box-shadow: 0 3px 10px rgba(37, 99, 235, 0.04);
    }
    div[data-testid="stTextInput"] input {
        background-color: #ffffff !important;
        color: #17233c !important;
        border: 1px solid #cbdff5 !important;
    }
    div[data-baseweb="select"] svg {
        color: #17233c !important;
    }
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    ul[role="listbox"] {
        background: #ffffff !important;
        color: #17233c !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 12px !important;
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.16) !important;
    }
    li[role="option"],
    div[role="option"] {
        background: #ffffff !important;
        color: #17233c !important;
    }
    li[role="option"]:hover,
    div[role="option"]:hover {
        background: #e8f4ff !important;
        color: #0f3557 !important;
    }
    li[aria-selected="true"],
    div[aria-selected="true"] {
        background: #dbeafe !important;
        color: #0f3557 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.88);
        border: 1.6px solid rgba(87, 156, 235, 0.62);
        border-radius: 18px;
        box-shadow: 0 10px 28px rgba(31, 87, 145, 0.10);
    }
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.88);
    }
    .hero-card {
        padding: 1.05rem 1.35rem;
        border-radius: 24px;
        background: linear-gradient(120deg, #e8f4ff 0%, #ebfff4 100%);
        border: 1px solid rgba(67, 151, 230, 0.18);
        box-shadow: 0 14px 36px rgba(44, 115, 181, 0.10);
        margin-bottom: 1.1rem;
    }
    .hero-card h1 {
        margin: 0 0 0.3rem 0 !important;
        color: #17233c;
    }
    .hero-caption {
        color: #53657d;
        font-size: 0.95rem;
    }
    .hero-badges {
        display: flex;
        gap: 0.55rem;
        flex-wrap: wrap;
        margin-top: 0.55rem;
    }
    .hero-badge {
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(82, 148, 226, 0.22);
        color: #24506f;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .section-label {
        color: #176b87;
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.03rem;
        margin-bottom: 0.15rem;
    }
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 0.75rem;
    }
    .section-title {
        color: #17233c;
        font-size: 1.02rem;
        font-weight: 800;
        white-space: nowrap;
    }
    .section-line {
        height: 2px;
        flex: 1;
        background: linear-gradient(90deg, #60a5fa 0%, rgba(96, 165, 250, 0.08) 100%);
    }
    .section-guide {
        color: #58708c;
        font-size: 0.82rem;
        margin-top: -0.35rem;
        margin-bottom: 0.55rem;
    }
    .field-label {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        color: #17233c;
        font-size: 0.86rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        position: relative;
    }
    .tooltip-dot {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1rem;
        height: 1rem;
        border-radius: 999px;
        background: #e0f2fe;
        border: 1px solid #60a5fa;
        color: #075985;
        font-size: 0.68rem;
        font-weight: 800;
        cursor: help;
    }
    .tooltip-text {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        z-index: 9999;
        left: 0;
        bottom: 1.45rem;
        width: 230px;
        padding: 0.55rem 0.65rem;
        border-radius: 10px;
        background: #17233c;
        color: #ffffff !important;
        font-size: 0.76rem;
        font-weight: 500;
        line-height: 1.35;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.18);
        transition: opacity 0.15s ease;
    }
    .field-label:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
    .inline-tooltip {
        position: relative;
        display: inline-flex;
        align-items: center;
        gap: 0.2rem;
        color: #24506f;
        font-weight: 700;
        border-bottom: 1px dotted #3b82f6;
        cursor: help;
    }
    .inline-tooltip .tooltip-dot {
        width: 0.9rem;
        height: 0.9rem;
        font-size: 0.62rem;
    }
    .inline-tooltip-text {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        z-index: 9999;
        left: 0;
        bottom: 1.4rem;
        width: 260px;
        padding: 0.55rem 0.65rem;
        border-radius: 10px;
        background: #17233c;
        color: #ffffff !important;
        font-size: 0.76rem;
        font-weight: 500;
        line-height: 1.35;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.18);
        transition: opacity 0.15s ease;
    }
    .inline-tooltip:hover .inline-tooltip-text {
        visibility: visible;
        opacity: 1;
    }
    .result-pill {
        border-radius: 16px;
        padding: 0.75rem 1rem;
        font-weight: 700;
        text-align: center;
        margin: 0.55rem 0 0.65rem 0;
    }
    .pill-low {
        background: #e8f8ef;
        color: #0f7a43;
        border: 1px solid #bdebd0;
    }
    .pill-high {
        background: #fff4e6;
        color: #b45f06;
        border: 1px solid #ffd49a;
    }
    .advice-box {
        border-radius: 16px;
        padding: 0.52rem 0.85rem 0.65rem 0.85rem;
        margin-top: 0.6rem;
        margin-bottom: 0.65rem;
        font-size: 0.88rem;
        line-height: 1.35;
    }
    .advice-low {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #14532d;
    }
    .advice-mid {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e3a8a;
    }
    .advice-high {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #7c2d12;
    }
    .small-note {
        color: #516174 !important;
        font-size: 0.78rem;
        line-height: 1.35;
        margin-top: 0.55rem;
        padding-top: 0.38rem;
        border-top: 1px solid rgba(96, 165, 250, 0.28);
    }
    .stButton > button {
        border-radius: 12px;
        border: 1px solid #93c5fd;
        background: linear-gradient(90deg, #e0f2fe 0%, #dcfce7 100%);
        color: #0f3557;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <h1>KNHANES 젊은 성인 고혈압 위험 예측</h1>
        <div class="hero-caption">
            교육용 머신러닝 데모 · 대상: 19-39세 · 기준:
            <span class="inline-tooltip">ACC/AHA 2017<span class="tooltip-dot">?</span>
                <span class="inline-tooltip-text">미국심장학회/미국심장협회에서 제시한 2017년 고혈압 기준입니다.</span>
            </span>,
            <span class="inline-tooltip">SBP<span class="tooltip-dot">?</span>
                <span class="inline-tooltip-text">수축기 혈압입니다. 혈압을 잴 때 위에 표시되는 숫자입니다.</span>
            </span>
            ≥ 130 또는
            <span class="inline-tooltip">DBP<span class="tooltip-dot">?</span>
                <span class="inline-tooltip-text">이완기 혈압입니다. 혈압을 잴 때 아래에 표시되는 숫자입니다.</span>
            </span>
            ≥ 80
        </div>
        <div class="hero-badges">
            <span class="hero-badge inline-tooltip">KNHANES 2017-2023
                <span class="tooltip-dot">?</span>
                <span class="inline-tooltip-text">국민건강영양조사 자료입니다. 본 프로젝트는 2017-2023년 자료를 사용했습니다.</span>
            </span>
            <span class="hero-badge">Young Adults 19-39</span>
            <span class="hero-badge inline-tooltip">post-COVID 기준 (2022-2023)
                <span class="tooltip-dot">?</span>
                <span class="inline-tooltip-text">KNHANES 측정체계가 통일된 코로나 이후(2022-2023) 환경을 기준으로 예측합니다.</span>
            </span>
            <span class="hero-badge inline-tooltip">PyCaret ML Model
                <span class="tooltip-dot">?</span>
                <span class="inline-tooltip-text">수업에서 사용한 자동화 머신러닝 라이브러리 기반 모델입니다.</span>
            </span>
            <span class="hero-badge">User Feedback Included</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([2.15, 1], gap="large")

with left:
    st.subheader("입력 정보")

    covid_period = "after"   # post-COVID(코로나 이후) 측정환경 기준 — 측정체계 일관 구간
    demo1, demo2 = st.columns([1, 1])
    with demo1:
        age = st.slider("나이", 19, 39, 25, help="만 나이 기준입니다. 본 프로젝트는 19-39세 젊은 성인을 대상으로 합니다.")
    with demo2:
        sex = st.selectbox("성별", options=[1.0, 2.0], format_func=lambda x: "남성" if x == 1.0 else "여성")

    with st.container(border=True):
        st.markdown('<div class="section-label">HEALTH CHECK</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header"><div class="section-title">건강검진 변수</div><div class="section-line"></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-guide">각 의학 용어 옆의 물음표에 마우스를 올리면 설명을 볼 수 있습니다.</div>', unsafe_allow_html=True)
        h1, h2, h3, h4, h5, h6 = st.columns(6)
        with h1:
            medical_label("BMI", "체질량지수입니다. 체중(kg)을 키(m)의 제곱으로 나눈 값입니다.")
            he_bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=60.0,
                value=23.0,
                step=0.1,
                label_visibility="collapsed",
            )
        with h2:
            medical_label("허리둘레(cm)", "복부비만과 관련된 신체계측 변수입니다.")
            he_wc = st.number_input(
                "허리둘레(cm)",
                min_value=50.0,
                max_value=150.0,
                value=80.0,
                step=0.1,
                label_visibility="collapsed",
            )
        with h3:
            medical_label("공복혈당", "공복 상태에서 측정한 혈당입니다.")
            he_glu = st.number_input(
                "공복혈당",
                min_value=50.0,
                max_value=300.0,
                value=95.0,
                step=1.0,
                label_visibility="collapsed",
            )
        with h4:
            medical_label("HbA1c", "당화혈색소입니다. 최근 2-3개월 평균 혈당 상태를 반영합니다.")
            he_hba1c = st.number_input(
                "HbA1c",
                min_value=3.0,
                max_value=15.0,
                value=5.3,
                step=0.1,
                label_visibility="collapsed",
            )
        with h5:
            medical_label("중성지방", "혈액 속 지방 성분 중 하나로, 대사 건강과 관련됩니다.")
            he_tg = st.number_input(
                "중성지방",
                min_value=20.0,
                max_value=800.0,
                value=120.0,
                step=1.0,
                label_visibility="collapsed",
            )
        with h6:
            medical_label("총콜레스테롤", "혈액 내 전체 콜레스테롤 수치입니다.")
            he_chol = st.number_input(
                "총콜레스테롤",
                min_value=80.0,
                max_value=400.0,
                value=180.0,
                step=1.0,
                label_visibility="collapsed",
            )

    with st.container(border=True):
        st.markdown('<div class="section-label">DIET VARIABLES</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header"><div class="section-title">식이 변수</div><div class="section-line"></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-guide">KNHANES 영양조사의 1일 섭취량 변수입니다. 잘 모르면 기본값을 사용해도 됩니다.</div>',
            unsafe_allow_html=True,
        )
        d1, d2 = st.columns(2)
        with d1:
            medical_label("나트륨 섭취량(mg/day)", "하루 동안 섭취한 나트륨 양입니다. 짠 음식 섭취 정도와 관련됩니다.")
            n_na = st.number_input(
                "나트륨 섭취량(mg/day)",
                min_value=0.0,
                max_value=15000.0,
                value=2967.0,
                step=100.0,
                label_visibility="collapsed",
            )
        with d2:
            medical_label("칼륨 섭취량(mg/day)", "하루 동안 섭취한 칼륨 양입니다. 채소·과일 섭취와 관련될 수 있습니다.")
            n_k = st.number_input(
                "칼륨 섭취량(mg/day)",
                min_value=0.0,
                max_value=10000.0,
                value=2350.0,
                step=100.0,
                label_visibility="collapsed",
            )

    with st.container(border=True):
        st.markdown('<div class="section-label">LIFESTYLE & SOCIOECONOMIC</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header"><div class="section-title">생활습관 및 사회경제 변수</div><div class="section-line"></div></div>',
            unsafe_allow_html=True,
        )
        l1, l2, l3, l4, l5 = st.columns([1, 1, 1, 1, 1.15])
        with l1:
            sm_presnt = st.selectbox("현재 흡연", options=[0.0, 1.0], format_func=lambda x: "아니오" if x == 0.0 else "예")
        with l2:
            dr_month = st.selectbox("월간 음주", options=[0.0, 1.0], format_func=lambda x: "아니오" if x == 0.0 else "예")
        with l3:
            pa_aerobic = st.selectbox("유산소 활동", options=[0.0, 1.0], format_func=lambda x: "아니오" if x == 0.0 else "예")
        with l4:
            medical_label(
                "가구소득 분위",
                "KNHANES에서 가구소득을 전체 조사대상자 기준으로 4개 그룹으로 나눈 값입니다. 정확한 금액이 아니라 상대적인 소득 위치를 의미합니다.",
            )
            incm = st.selectbox(
                "가구소득 분위",
                options=[1.0, 2.0, 3.0, 4.0],
                format_func=lambda x: income_labels[x],
                label_visibility="collapsed",
            )
        with l5:
            medical_label("교육 수준", "최종 학력 또는 교육 이수 수준을 나타내는 변수입니다.")
            educ = st.selectbox(
                "교육 수준",
                options=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                format_func=lambda x: education_labels[x],
                label_visibility="collapsed",
            )

    with st.container(border=True):
        st.markdown('<div class="section-label">FAMILY HISTORY & STRESS</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header"><div class="section-title">가족력 및 스트레스</div><div class="section-line"></div></div>',
            unsafe_allow_html=True,
        )
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            medical_label("부 가족력", "아버지의 고혈압 가족력 여부입니다.")
            he_hpfh1 = st.selectbox(
                "부 가족력",
                options=[0.0, 1.0],
                format_func=lambda x: "없음" if x == 0.0 else "있음",
                label_visibility="collapsed",
            )
        with f2:
            medical_label("모 가족력", "어머니의 고혈압 가족력 여부입니다.")
            he_hpfh2 = st.selectbox(
                "모 가족력",
                options=[0.0, 1.0],
                format_func=lambda x: "없음" if x == 0.0 else "있음",
                label_visibility="collapsed",
            )
        with f3:
            medical_label("형제자매 가족력", "형제자매의 고혈압 가족력 여부입니다.")
            he_hpfh3 = st.selectbox(
                "형제자매 가족력",
                options=[0.0, 1.0],
                format_func=lambda x: "없음" if x == 0.0 else "있음",
                label_visibility="collapsed",
            )
        with f4:
            medical_label("스트레스 인지", "평소 스트레스를 많이 느끼는지에 대한 자기보고 변수입니다.")
            mh_stress = st.selectbox(
                "스트레스 인지",
                options=[0.0, 1.0],
                format_func=lambda x: "낮음" if x == 0.0 else "높음",
                label_visibility="collapsed",
            )


input_data = pd.DataFrame(
    [
        {
            "age": age,
            "HE_BMI": he_bmi,
            "HE_wc": he_wc,
            "HE_glu": he_glu,
            "HE_HbA1c": he_hba1c,
            "HE_TG": he_tg,
            "HE_chol": he_chol,
            "N_NA": n_na,
            "N_K": n_k,
            "sex": sex,
            "sm_presnt": sm_presnt,
            "dr_month": dr_month,
            "pa_aerobic": pa_aerobic,
            "incm": incm,
            "educ": educ,
            "HE_HPfh1": he_hpfh1,
            "HE_HPfh2": he_hpfh2,
            "HE_HPfh3": he_hpfh3,
            "mh_stress": mh_stress,
            "covid_period": covid_period,
        }
    ]
)


with right:
    st.subheader("실시간 예측")
    with st.container(border=True):
        prediction = predict_model(model, data=input_data, raw_score=True)
        pred = int(prediction.loc[0, "prediction_label"])
        if "prediction_score_1" in prediction.columns:
            prob = float(prediction.loc[0, "prediction_score_1"])
        else:
            prob = float(prediction.loc[0, "prediction_score"])

        st.metric("고혈압 위험 예측 확률", f"{prob:.1%}")

        if pred == 1:
            st.markdown(
                '<div class="result-pill pill-high">모델 예측: 고혈압 위험군</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-pill pill-low">모델 예측: 비고혈압군</div>',
                unsafe_allow_html=True,
            )

        st.progress(min(max(prob, 0.0), 1.0))
        if prob >= 0.6:
            advice_class = "advice-high"
            advice_title = "병원 방문 권장"
            advice_text = "가까운 병원이나 보건소에서 실제 혈압 측정과 상담을 권장합니다."
        elif prob >= 0.4:
            advice_class = "advice-mid"
            advice_title = "혈압 확인 권장"
            advice_text = "최근 혈압 측정 경험이 없다면 자가 측정 또는 병원/보건소 확인을 권장합니다."
        else:
            advice_class = "advice-low"
            advice_title = "생활습관 유지 및 정기 확인"
            advice_text = "정기적인 혈압 확인과 생활습관 관리는 계속 필요합니다."

        st.markdown(
            f"""
            <div class="advice-box {advice_class}">
                <strong>{advice_title}</strong><br>
                {advice_text}
                <div class="small-note">※ 본 결과는 교육용 예측입니다. 의학적 판단은 전문가 진단이 필요합니다.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("해석 시 주의점", expanded=False):
        st.write("조사 시기 변수는 모델 학습 과정의 데이터 구분 변수로 사용되었으며, 사용자 입력 항목에서는 제외했습니다.")
        st.write("현재 앱에서는 최근 측정환경에 가까운 `after` 값을 기본으로 적용합니다.")
        st.write("HDL은 측정 시기별 변화 가능성, LDL은 높은 결측률 때문에 주분석 모델 feature에서 제외했습니다.")

    st.subheader("사용자 피드백")
    helpful = st.radio("예측 결과가 도움이 되었나요?", ["예", "아니오"], horizontal=True)
    comment = st.text_input("추가 의견", placeholder="추가 의견(선택)", label_visibility="collapsed")
    st.caption("피드백은 예측 결과와 함께 로컬 CSV 파일에 저장되어 앱 개선 방향을 검토하는 데 사용됩니다.")

    if st.button("피드백 저장", use_container_width=True):
        FEEDBACK_PATH.parent.mkdir(exist_ok=True)
        feedback_row = pd.DataFrame(
            [
                {
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "prediction_label": pred,
                    "prediction_probability": round(prob, 4),
                    "helpful": helpful,
                    "comment": comment,
                }
            ]
        )

        if FEEDBACK_PATH.exists():
            old = pd.read_csv(FEEDBACK_PATH)
            feedback = pd.concat([old, feedback_row], ignore_index=True)
        else:
            feedback = feedback_row

        feedback.to_csv(FEEDBACK_PATH, index=False, encoding="utf-8-sig")
        st.success("피드백이 저장되었습니다.")

    if FEEDBACK_PATH.exists():
        feedback_preview = pd.read_csv(FEEDBACK_PATH)
        if "helpful" in feedback_preview.columns and len(feedback_preview) > 0:
            helpful_rate = (feedback_preview["helpful"] == "예").mean() * 100
            st.caption(f"현재 저장된 피드백 {len(feedback_preview)}건 중 도움이 되었다는 응답은 {helpful_rate:.1f}%입니다.")

