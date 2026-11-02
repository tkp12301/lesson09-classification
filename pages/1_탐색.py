
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="뇌졸중 데이터 탐색",
    page_icon="🔎",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/tkp12301/data/refs/heads/main/stroke.csv"


# 데이터 불러오기
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🔎 뇌졸중 데이터 탐색")
st.write("뇌졸중 데이터의 분포와 주요 변수 사이의 관계를 탐색합니다.")

try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(e))
    st.stop()


# 숫자형 데이터 변환
numeric_columns = [
    "age",
    "avg_glucose_level",
    "bmi",
    "stroke",
    "hypertension",
    "heart_disease"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 뇌졸중 값이 확인된 데이터
valid_df = df[df["stroke"].isin([0, 1])].copy()

valid_df["뇌졸중 여부"] = valid_df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})


# ==================================================
# 1. 나이와 평균 혈당 분포
# ==================================================

st.divider()
st.header("1. 나이와 평균 혈당의 분포")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df.dropna(subset=["age"]),
        x="age",
        nbins=30,
        title="나이 분포"
    )

    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수"
    )

    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_glucose = px.histogram(
        df.dropna(subset=["avg_glucose_level"]),
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포"
    )

    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수"
    )

    st.plotly_chart(fig_glucose, use_container_width=True)


# ==================================================
# 2. 뇌졸중 유무에 따른 나이와 혈당 비교
# ==================================================

st.divider()
st.header("2. 뇌졸중 유무에 따른 특성 비교")

col1, col2 = st.columns(2)

with col1:
    age_box = valid_df.dropna(subset=["age"])

    fig_age_box = px.box(
        age_box,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        category_orders={
            "뇌졸중 여부": ["뇌졸중 없음", "뇌졸중 있음"]
        },
        title="뇌졸중 유무에 따른 나이 비교"
    )

    fig_age_box.update_layout(
        showlegend=False,
        xaxis_title="뇌졸중 여부",
        yaxis_title="나이"
    )

    st.plotly_chart(fig_age_box, use_container_width=True)

with col2:
    glucose_box = valid_df.dropna(
        subset=["avg_glucose_level"]
    )

    fig_glucose_box = px.box(
        glucose_box,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        category_orders={
            "뇌졸중 여부": ["뇌졸중 없음", "뇌졸중 있음"]
        },
        title="뇌졸중 유무에 따른 평균 혈당 비교"
    )

    fig_glucose_box.update_layout(
        showlegend=False,
        xaxis_title="뇌졸중 여부",
        yaxis_title="평균 혈당"
    )

    st.plotly_chart(fig_glucose_box, use_container_width=True)


# 평균값 표
mean_rows = []

for group in ["뇌졸중 없음", "뇌졸중 있음"]:

    group_df = valid_df[
        valid_df["뇌졸중 여부"] == group
    ]

    mean_rows.append({
        "뇌졸중 여부": group,
        "사람 수": len(group_df),
        "평균 나이": group_df["age"].mean(),
        "평균 혈당": group_df["avg_glucose_level"].mean()
    })

mean_table = pd.DataFrame(mean_rows)

mean_table["평균 나이"] = mean_table["평균 나이"].round(2)
mean_table["평균 혈당"] = mean_table["평균 혈당"].round(2)

st.subheader("두 그룹의 평균값 비교")
st.dataframe(mean_table, use_container_width=True, hide_index=True)


# ==================================================
# 공통 함수: 뇌졸중 비율 계산
# ==================================================

def calculate_rate(data, column):

    result = []

    for value in [0, 1]:

        group = data[data[column] == value]

        total = len(group)

        stroke_count = int(
            (group["stroke"] == 1).sum()
        )

        rate = (
            stroke_count / total * 100
            if total > 0 else 0
        )

        result.append({
            "구분값": value,
            "전체 사람 수": total,
            "뇌졸중 사람 수": stroke_count,
            "뇌졸중 비율(%)": round(rate, 2)
        })

    return pd.DataFrame(result)


def draw_rate_chart(data, column, labels, title):

    result = calculate_rate(data, column)

    result["구분"] = result["구분값"].map(labels)

    fig = px.bar(
        result,
        x="구분",
        y="뇌졸중 비율(%)",
        text="뇌졸중 비율(%)",
        title=title
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="구분",
        yaxis_title="뇌졸중 비율(%)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        result[
            [
                "구분",
                "전체 사람 수",
                "뇌졸중 사람 수",
                "뇌졸중 비율(%)"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# 3. 고혈압과 뇌졸중
# ==================================================

st.divider()
st.header("3. 고혈압 유무에 따른 뇌졸중 비율")

draw_rate_chart(
    valid_df,
    "hypertension",
    {
        0: "고혈압 없음",
        1: "고혈압 있음"
    },
    "고혈압 유무에 따른 뇌졸중 비율"
)


# ==================================================
# 4. 심장병과 뇌졸중
# ==================================================

st.divider()
st.header("4. 심장병 유무에 따른 뇌졸중 비율")

draw_rate_chart(
    valid_df,
    "heart_disease",
    {
        0: "심장병 없음",
        1: "심장병 있음"
    },
    "심장병 유무에 따른 뇌졸중 비율"
)


# ==================================================
# 5. BMI 빈 값과 뇌졸중 비율
# ==================================================

st.divider()
st.header("5. BMI 빈 값과 뇌졸중 비율")

missing_bmi = valid_df[
    valid_df["bmi"].isna()
]

# 전체 데이터
total_people = len(valid_df)

total_stroke = int(
    (valid_df["stroke"] == 1).sum()
)

total_rate = (
    total_stroke / total_people * 100
    if total_people > 0 else 0
)

# BMI 빈 값인 사람
missing_people = len(missing_bmi)

missing_stroke = int(
    (missing_bmi["stroke"] == 1).sum()
)

missing_rate = (
    missing_stroke / missing_people * 100
    if missing_people > 0 else 0
)

bmi_table = pd.DataFrame([
    {
        "구분": "BMI 빈 값인 사람",
        "사람 수": missing_people,
        "뇌졸중 사람 수": missing_stroke,
        "뇌졸중 비율(%)": round(missing_rate, 2)
    },
    {
        "구분": "전체 사람",
        "사람 수": total_people,
        "뇌졸중 사람 수": total_stroke,
        "뇌졸중 비율(%)": round(total_rate, 2)
    }
])

st.dataframe(
    bmi_table,
    use_container_width=True,
    hide_index=True
)


# ==================================================
# 6. 흡연 상태별 사람 수
# ==================================================

st.divider()
st.header("6. 흡연 상태별 사람 수")

smoking_table = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .reset_index()
)

smoking_table.columns = [
    "흡연 상태",
    "사람 수"
]

smoking_table["흡연 상태"] = (
    smoking_table["흡연 상태"]
    .fillna("빈 값")
)

st.dataframe(
    smoking_table,
    use_container_width=True,
    hide_index=True
)


# ==================================================
# 마무리
# ==================================================

st.divider()
st.subheader("탐색 결과 해석 시 주의사항")

st.info(
    "이 분석은 데이터에서 나타나는 통계적 관계를 보여 줍니다. "
    "고혈압이나 심장병이 뇌졸중의 직접적인 원인이라고 "
    "단정할 수는 없습니다. "
    "BMI 빈 값 그룹의 비율 차이 역시 인과관계를 의미하지 않습니다."
)
