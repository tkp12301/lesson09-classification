
import pandas as pd
import streamlit as st
import plotly.express as px

# 페이지 설정
st.set_page_config(
    page_title="뇌졸중 데이터 탐색",
    page_icon="🔎",
    layout="wide"
)

# 데이터 주소
DATA_URL = "https://raw.githubusercontent.com/tkp12301/data/refs/heads/main/stroke.csv"


# 데이터 불러오기
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🔎 뇌졸중 데이터 탐색")
st.write(
    "데이터의 분포와 뇌졸중 발생 사이의 관계를 "
    "그래프와 표를 이용해 탐색합니다."
)

try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.write("인터넷 연결과 CSV 주소를 확인해 주세요.")
    st.code(str(e))
    st.stop()


# 숫자형 데이터 변환
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["avg_glucose_level"] = pd.to_numeric(
    df["avg_glucose_level"], errors="coerce"
)
df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
df["stroke"] = pd.to_numeric(df["stroke"], errors="coerce")

# 뇌졸중 그룹 이름 생성
df["stroke_group"] = df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})

# 분석 가능한 데이터
valid_stroke = df[df["stroke"].isin([0, 1])].copy()


# ==================================================
# 1. 나이와 평균 혈당 분포
# ==================================================

st.divider()
st.header("1. 나이와 평균 혈당의 분포")

st.write(
    "히스토그램을 이용해 전체 데이터에서 "
    "나이와 평균 혈당이 어떤 형태로 분포하는지 확인합니다."
)

col1, col2 = st.columns(2)

with col1:
    age_data = df["age"].dropna()

    fig_age = px.histogram(
        age_data,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={"age": "나이", "count": "사람 수"},
    )

    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수",
        bargap=0.05
    )

    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    glucose_data = df["avg_glucose_level"].dropna()

    fig_glucose = px.histogram(
        glucose_data,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수"
        },
    )

    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수",
        bargap=0.05
    )

    st.plotly_chart(fig_glucose, use_container_width=True)


# ==================================================
# 2. 뇌졸중 유무에 따른 나이와 평균 혈당 비교
# ==================================================

st.divider()
st.header("2. 뇌졸중 유무에 따른 특성 비교")

st.write(
    "뇌졸중을 겪은 사람과 겪지 않은 사람으로 나누어 "
    "나이와 평균 혈당의 분포를 상자그림으로 비교합니다."
)

col1, col2 = st.columns(2)

with col1:
    age_box_data = valid_stroke.dropna(
        subset=["age", "stroke_group"]
    )

    fig_age_box = px.box(
        age_box_data,
        x="stroke_group",
        y="age",
        color="stroke_group",
        title="뇌졸중 유무에 따른 나이 비교",
        labels={
            "stroke_group": "뇌졸중 여부",
            "age": "나이"
        },
        category_orders={
            "stroke_group": ["뇌졸중 없음", "뇌졸중 있음"]
        }
    )

    fig_age_box.update_layout(
        showlegend=False,
        xaxis_title="뇌졸중 여부",
        yaxis_title="나이"
    )

    st.plotly_chart(fig_age_box, use_container_width=True)

with col2:
    glucose_box_data = valid_stroke.dropna(
        subset=["avg_glucose_level", "stroke_group"]
    )

    fig_glucose_box = px.box(
        glucose_box_data,
        x="stroke_group",
        y="avg_glucose_level",
        color="stroke_group",
        title="뇌졸중 유무에 따른 평균 혈당 비교",
        labels={
            "stroke_group": "뇌졸중 여부",
            "avg_glucose_level": "평균 혈당"
        },
        category_orders={
            "stroke_group": ["뇌졸중 없음", "뇌졸중 있음"]
        }
    )

    fig_glucose_box.update_layout(
        showlegend=False,
        xaxis_title="뇌졸중 여부",
        yaxis_title="평균 혈당"
    )

    st.plotly_chart(fig_glucose_box, use_container_width=True)


# 두 그룹의 평균값 계산
mean_data = valid_stroke.groupby(
    "stroke_group",
    observed=False
).agg(
    사람 수=("stroke", "size"),
    평균 나이=("age", "mean"),
    평균 혈당=("avg_glucose_level", "mean")
).reindex(["뇌졸중 없음", "뇌졸중 있음"])

mean_data = mean_data.reset_index()

mean_data["평균 나이"] = mean_data["평균 나이"].round(2)
mean_data["평균 혈당"] = mean_data["평균 혈당"].round(2)

st.subheader("뇌졸중 유무에 따른 평균값")

st.dataframe(
    mean_data,
    use_container_width=True,
    hide_index=True
)


# ==================================================
# 3. 고혈압 유무에 따른 뇌졸중 비율
# ==================================================

st.divider()
st.header("3. 고혈압과 뇌졸중의 관계")

st.write(
    "고혈압이 있는 사람과 없는 사람을 구분하여 "
    "각 그룹의 뇌졸중 발생 비율을 비교합니다."
)


def calculate_stroke_rate(data, column):
    result = (
        data[data[column].isin([0, 1])]
        .groupby(column)
        .agg(
            전체_사람=("stroke", "size"),
            뇌졸중_사람=("stroke", "sum")
        )
        .reset_index()
    )

    result["뇌졸중 비율"] = (
        result["뇌졸중_사람"] /
        result["전체_사람"] * 100
    )

    return result


hypertension_data = calculate_stroke_rate(
    valid_stroke, "hypertension"
)

hypertension_data["고혈압 여부"] = (
    hypertension_data["hypertension"]
    .map({0: "고혈압 없음", 1: "고혈압 있음"})
)

fig_hypertension = px.bar(
    hypertension_data,
    x="고혈압 여부",
    y="뇌졸중 비율",
    text="뇌졸중 비율",
    title="고혈압 유무에 따른 뇌졸중 비율",
    labels={
        "고혈압 여부": "고혈압 여부",
        "뇌졸중 비율": "뇌졸중 비율(%)"
    }
)

fig_hypertension.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig_hypertension.update_layout(
    yaxis_title="뇌졸중 비율(%)",
    xaxis_title="고혈압 여부"
)

st.plotly_chart(
    fig_hypertension,
    use_container_width=True
)


# ==================================================
# 4. 심장병 유무에 따른 뇌졸중 비율
# ==================================================

st.divider()
st.header("4. 심장병과 뇌졸중의 관계")

st.write(
    "심장병이 있는 사람과 없는 사람의 "
    "뇌졸중 발생 비율을 비교합니다."
)

heart_data = calculate_stroke_rate(
    valid_stroke, "heart_disease"
)

heart_data["심장병 여부"] = (
    heart_data["heart_disease"]
    .map({0: "심장병 없음", 1: "심장병 있음"})
)

fig_heart = px.bar(
    heart_data,
    x="심장병 여부",
    y="뇌졸중 비율",
    text="뇌졸중 비율",
    title="심장병 유무에 따른 뇌졸중 비율",
    labels={
        "심장병 여부": "심장병 여부",
        "뇌졸중 비율": "뇌졸중 비율(%)"
    }
)

fig_heart.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig_heart.update_layout(
    yaxis_title="뇌졸중 비율(%)",
    xaxis_title="심장병 여부"
)

st.plotly_chart(
    fig_heart,
    use_container_width=True
)


# ==================================================
# 5. BMI 결측치와 뇌졸중 비율
# ==================================================

st.divider()
st.header("5. BMI 빈 값과 뇌졸중의 관계")

st.write(
    "BMI 값이 비어 있는 사람들을 따로 구분하고 "
    "이들의 뇌졸중 비율을 전체 데이터의 비율과 비교합니다."
)

# BMI가 비어 있는 사람
missing_bmi = valid_stroke[valid_stroke["bmi"].isna()]

# 전체 데이터에서 뇌졸중 비율
total_valid_count = len(valid_stroke)
total_stroke_count = int(valid_stroke["stroke"].sum())

total_rate = (
    total_stroke_count / total_valid_count * 100
    if total_valid_count else 0
)

# BMI 결측 그룹의 뇌졸중 비율
missing_count = len(missing_bmi)
missing_stroke_count = int(missing_bmi["stroke"].sum())

missing_rate = (
    missing_stroke_count / missing_count * 100
    if missing_count else 0
)

bmi_comparison = pd.DataFrame({
    "구분": ["BMI 빈 값인 사람", "전체 사람"],
    "사람 수": [missing_count, total_valid_count],
    "뇌졸중 사람 수": [
        missing_stroke_count,
        total_stroke_count
    ],
    "뇌졸중 비율(%)": [
        round(missing_rate, 2),
        round(total_rate, 2)
    ]
})

st.dataframe(
    bmi_comparison,
    use_container_width=True,
    hide_index=True
)


# ==================================================
# 6. 흡연 상태별 사람 수
# ==================================================

st.divider()
st.header("6. 흡연 상태별 사람 수")

st.write(
    "흡연 상태의 각 값에 해당하는 사람 수를 세어 "
    "표로 나타냅니다. 빈 값도 별도로 표시합니다."
)

smoking_counts = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

smoking_counts["흡연 상태"] = (
    smoking_counts["흡연 상태"]
    .fillna("빈 값")
)

st.dataframe(
    smoking_counts,
    use_container_width=True,
    hide_index=True
)


# ==================================================
# 마무리
# ==================================================

st.divider()

st.subheader("탐색 결과 해석 시 주의사항")

st.info(
    "그래프와 표는 데이터에 나타난 통계적 관계를 보여 줍니다. "
    "고혈압이나 심장병이 뇌졸중의 직접적인 원인이라고 "
    "이 결과만으로 단정할 수는 없습니다. "
    "또한 BMI 빈 값 그룹의 비율 차이만으로 "
    "결측 여부와 뇌졸중 사이의 인과관계를 판단할 수 없습니다."
)
