
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 1. 기본 설정 및 데이터 불러오기
# --------------------------------------------------

st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 뇌졸중 예측 실습실")
st.write(
    "데이터를 전처리한 뒤 로지스틱 회귀와 의사결정트리를 학습하고 "
    "두 모델의 예측 결과를 비교합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "tkp12301/data/refs/heads/main/stroke.csv"
)

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")
    return data


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# 2. 분석할 속성 선택
# --------------------------------------------------

st.header("1. 분석할 속성 선택")

st.write(
    "뇌졸중 예측에 사용할 속성을 선택하세요. "
    "초기 설정은 나이, 평균 혈당, 고혈압, 심장병입니다."
)

selected_features = st.multiselect(
    "예측에 사용할 속성",
    options=list(FEATURE_NAMES.keys()),
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease"
    ],
    format_func=lambda x: FEATURE_NAMES[x]
)

if len(selected_features) < 2:
    st.warning("분류 모델을 학습하려면 속성을 최소 2개 선택해야 합니다.")
    st.stop()

st.write(
    "**선택한 속성:** "
    + ", ".join(FEATURE_NAMES[x] for x in selected_features)
)


# --------------------------------------------------
# 3. 데이터 기본 정리
# --------------------------------------------------

st.header("2. 데이터 전처리")

st.subheader("① 데이터 기본 확인")

st.write("원본 데이터의 크기와 결측값을 확인합니다.")

st.write(f"- 원본 데이터 행 수: **{len(df):,}명**")
st.write(f"- 원본 데이터 열 수: **{len(df.columns)}개**")

required_columns = ["id", "stroke"] + selected_features

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(
        "데이터에 필요한 열이 없습니다: "
        + ", ".join(missing_columns)
    )
    st.stop()

# ID와 예측 대상 숫자형 변환
df["id"] = pd.to_numeric(df["id"], errors="coerce")
df["stroke"] = pd.to_numeric(df["stroke"], errors="coerce")

# 예측 대상이 0 또는 1인 데이터만 사용
df = df.dropna(subset=["id", "stroke"])
df = df[df["stroke"].isin([0, 1])].copy()

df["stroke"] = df["stroke"].astype(int)

# ID 번호 순서로 정렬
df = df.sort_values(
    by="id",
    kind="mergesort"
).reset_index(drop=True)

# 선택된 속성을 숫자형으로 변환
for col in selected_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

st.write("**선택한 속성의 결측값 현황**")

missing_table = pd.DataFrame({
    "속성": [FEATURE_NAMES[col] for col in selected_features],
    "결측값 수": [int(df[col].isna().sum()) for col in selected_features]
})

st.dataframe(missing_table, use_container_width=True)


# --------------------------------------------------
# 4. 훈련용 / 테스트용 데이터 분리
# --------------------------------------------------

st.subheader("② 훈련용 데이터와 테스트용 데이터 분리")

st.write(
    "ID 순서로 정렬한 뒤 10명씩 묶어 각 묶음의 앞 3명은 "
    "테스트용으로, 나머지 7명은 훈련용으로 분리합니다."
)

# 10명 단위로 앞의 3명을 테스트 데이터로 지정
test_mask = [
    i % 10 < 3 for i in range(len(df))
]

train_df = df.loc[[not x for x in test_mask]].copy()
test_df = df.loc[test_mask].copy()

st.write(f"- 전체 분석 데이터: **{len(df):,}명**")
st.write(f"- 훈련용 데이터: **{len(train_df):,}명**")
st.write(f"- 테스트용 데이터: **{len(test_df):,}명**")

if len(train_df) == 0 or len(test_df) == 0:
    st.error("훈련용 또는 테스트용 데이터가 없습니다.")
    st.stop()

if train_df["stroke"].nunique() < 2:
    st.error(
        "훈련용 데이터에 뇌졸중 여부가 한 종류만 존재합니다. "
        "두 종류의 결과가 있어야 분류 모델을 학습할 수 있습니다."
    )
    st.stop()

st.write("**데이터 분리 결과**")

split_table = pd.DataFrame({
    "구분": ["훈련용 데이터", "테스트용 데이터"],
    "인원": [len(train_df), len(test_df)],
    "뇌졸중 환자 수": [
        int(train_df["stroke"].sum()),
        int(test_df["stroke"].sum())
    ],
    "뇌졸중 비율(%)": [
        round(train_df["stroke"].mean() * 100, 2),
        round(test_df["stroke"].mean() * 100, 2)
    ]
})

st.dataframe(split_table, use_container_width=True)


# --------------------------------------------------
# 5. 결측값 처리
# --------------------------------------------------

st.subheader("③ 결측값 처리")

st.write(
    "BMI를 선택한 경우 훈련용 데이터의 중앙값을 계산해 "
    "훈련용과 테스트용 데이터의 BMI 결측값을 채웁니다. "
    "테스트용 데이터의 정보를 이용해 중앙값을 계산하지 않습니다."
)

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].copy()
y_test = test_df["stroke"].copy()

preprocessing_notes = []

if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()

    if pd.isna(bmi_median):
        st.error("훈련용 BMI 데이터에 유효한 값이 없습니다.")
        st.stop()

    train_bmi_missing = int(X_train["bmi"].isna().sum())
    test_bmi_missing = int(X_test["bmi"].isna().sum())

    X_train.loc[:, "bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test.loc[:, "bmi"] = X_test["bmi"].fillna(bmi_median)

    preprocessing_notes.append({
        "처리 속성": "체질량지수(BMI)",
        "처리 방법": "훈련용 데이터 중앙값으로 결측값 대체",
        "사용한 중앙값": round(float(bmi_median), 2),
        "훈련용 결측값 처리 수": train_bmi_missing,
        "테스트용 결측값 처리 수": test_bmi_missing
    })

# BMI 이외의 선택 속성에 결측값이 있는 경우
# 훈련용 데이터에서 계산한 중앙값으로 처리
for col in selected_features:
    if col == "bmi":
        continue

    if X_train[col].isna().any() or X_test[col].isna().any():
        median_value = X_train[col].median()

        if pd.isna(median_value):
            st.error(
                f"{FEATURE_NAMES[col]} 속성에 사용할 수 있는 "
                "훈련용 데이터가 없습니다."
            )
            st.stop()

        train_missing = int(X_train[col].isna().sum())
        test_missing = int(X_test[col].isna().sum())

        X_train.loc[:, col] = X_train[col].fillna(median_value)
        X_test.loc[:, col] = X_test[col].fillna(median_value)

        preprocessing_notes.append({
            "처리 속성": FEATURE_NAMES[col],
            "처리 방법": "훈련용 데이터 중앙값으로 결측값 대체",
            "사용한 중앙값": round(float(median_value), 2),
            "훈련용 결측값 처리 수": train_missing,
            "테스트용 결측값 처리 수": test_missing
        })

if preprocessing_notes:
    st.dataframe(
        pd.DataFrame(preprocessing_notes),
        use_container_width=True
    )
else:
    st.info("선택한 속성에 처리할 결측값이 없습니다.")

st.write("**전처리 완료 후 데이터 확인**")

st.write(
    f"- 훈련용 데이터 결측값: **{int(X_train.isna().sum().sum())}개**"
)
st.write(
    f"- 테스트용 데이터 결측값: **{int(X_test.isna().sum().sum())}개**"
)

if X_train.isna().any().any() or X_test.isna().any().any():
    st.error("전처리 후에도 결측값이 남아 있습니다.")
    st.stop()


# --------------------------------------------------
# 6. 로지스틱 회귀를 위한 표준화
# --------------------------------------------------

st.subheader("④ 데이터 크기 맞추기(표준화)")

st.write(
    "로지스틱 회귀는 속성마다 값의 범위가 다르면 학습에 영향을 "
    "받을 수 있으므로 훈련용 데이터의 평균과 표준편차를 이용해 "
    "표준화합니다. 의사결정트리는 표준화하지 않은 원래 값을 사용합니다."
)

train_mean = X_train.mean()
train_std = X_train.std()
train_std = train_std.replace(0, 1).fillna(1)

X_train_scaled = (X_train - train_mean) / train_std
X_test_scaled = (X_test - train_mean) / train_std

scale_table = pd.DataFrame({
    "속성": [FEATURE_NAMES[col] for col in selected_features],
    "훈련용 평균": [
        round(float(train_mean[col]), 3)
        for col in selected_features
    ],
    "훈련용 표준편차": [
        round(float(train_std[col]), 3)
        for col in selected_features
    ]
})

st.dataframe(scale_table, use_container_width=True)

st.write("**표준화된 훈련용 데이터 일부**")
st.dataframe(
    X_train_scaled.head(),
    use_container_width=True
)


# --------------------------------------------------
# 7. 분류 모델 학습
# --------------------------------------------------

st.header("3. 분류 모델 학습")

st.write(
    "전처리한 훈련용 데이터만 사용하여 두 가지 분류 모델을 학습합니다. "
    "테스트용 데이터는 학습에 사용하지 않고 모델의 성능을 확인할 때 사용합니다."
)

# 로지스틱 회귀
logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

logistic_model.fit(X_train_scaled, y_train)

# 의사결정트리
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(X_train, y_train)

st.subheader("① 학습 결과")

training_table = pd.DataFrame({
    "모델": [
        "로지스틱 회귀(확률로 답하는 모델)",
        "의사결정트리(질문으로 답하는 모델)"
    ],
    "학습 데이터": [
        len(X_train_scaled),
        len(X_train)
    ],
    "사용한 속성 수": [
        len(selected_features),
        len(selected_features)
    ],
    "학습에 사용한 데이터": [
        "표준화된 훈련용 데이터",
        "전처리된 훈련용 데이터"
    ]
})

st.dataframe(training_table, use_container_width=True)

st.write("**모델별 설정**")
st.write(
    "- 로지스틱 회귀: 반복 학습 횟수 2,000회, 난수 고정값 42"
)
st.write(
    "- 의사결정트리: 최대 질문 횟수 3회, "
    "마지막 마디 최소 인원 5명, 난수 고정값 42"
)


# --------------------------------------------------
# 8. 정확도 계산 및 기준 모델
# --------------------------------------------------

st.subheader("② 모델 성능 평가")

# 로지스틱 회귀 예측
logistic_train_pred = logistic_model.predict(X_train_scaled)
logistic_test_pred = logistic_model.predict(X_test_scaled)

# 의사결정트리 예측
tree_train_pred = tree_model.predict(X_train)
tree_test_pred = tree_model.predict(X_test)

# 훈련용 데이터에서 가장 많은 결과만 답하는 기준 모델
majority_class = int(y_train.value_counts().idxmax())

baseline_train_pred = [majority_class] * len(y_train)
baseline_test_pred = [majority_class] * len(y_test)

def accuracy(actual, predicted):
    return float((actual.reset_index(drop=True) ==
                  pd.Series(predicted).reset_index(drop=True)).mean())


logistic_train_accuracy = accuracy(y_train, logistic_train_pred)
logistic_test_accuracy = accuracy(y_test, logistic_test_pred)

tree_train_accuracy = accuracy(y_train, tree_train_pred)
tree_test_accuracy = accuracy(y_test, tree_test_pred)

baseline_train_accuracy = accuracy(y_train, baseline_train_pred)
baseline_test_accuracy = accuracy(y_test, baseline_test_pred)


# 정확도 카드
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "로지스틱 회귀",
        f"{logistic_test_accuracy * 100:.2f}%"
    )
    st.caption(
        f"훈련용 정확도: {logistic_train_accuracy * 100:.2f}%"
    )
    st.caption(
        f"테스트용 정확도: {logistic_test_accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "의사결정트리",
        f"{tree_test_accuracy * 100:.2f}%"
    )
    st.caption(
        f"훈련용 정확도: {tree_train_accuracy * 100:.2f}%"
    )
    st.caption(
        f"테스트용 정확도: {tree_test_accuracy * 100:.2f}%"
    )

with col3:
    st.metric(
        "기준 모델",
        f"{baseline_test_accuracy * 100:.2f}%"
    )
    st.caption(
        f"훈련용 정확도: {baseline_train_accuracy * 100:.2f}%"
    )
    st.caption(
        f"테스트용 정확도: {baseline_test_accuracy * 100:.2f}%"
    )

st.write(
    "기준 모델은 속성을 전혀 확인하지 않고 훈련용 데이터에서 "
    "더 많은 쪽의 결과만 계속 예측하는 모델입니다."
)

if majority_class == 1:
    st.write("기준 모델의 예측 결과: 항상 뇌졸중이라고 예측")
else:
    st.write("기준 모델의 예측 결과: 항상 뇌졸중이 아니라고 예측")


# --------------------------------------------------
# 9. 그래프에 사용할 x축, y축 선택
# --------------------------------------------------

st.header("4. 분류 결과 시각화")

st.write(
    "선택한 두 속성을 x축과 y축으로 설정해 테스트 데이터의 "
    "실제 결과와 모델이 나눈 예측 영역을 비교합니다."
)

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x]
    )

with axis_col2:
    y_options = [
        col for col in selected_features if col != x_feature
    ]

    y_feature = st.selectbox(
        "세로축 속성",
        options=y_options,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x]
    )

# 그래프에서 사용하지 않는 나머지 속성은 테스트 데이터 중앙값으로 고정
fixed_values = {}

for col in selected_features:
    if col not in [x_feature, y_feature]:
        fixed_values[col] = float(X_test[col].median())

if fixed_values:
    fixed_text = ", ".join([
        f"{FEATURE_NAMES[col]}: {value:.2f}"
        for col, value in fixed_values.items()
    ])
    st.info(
        "그래프의 나머지 속성은 테스트 데이터 중앙값으로 고정했습니다. "
        + fixed_text
    )


# --------------------------------------------------
# 10. 그래프용 격자 데이터 생성
# --------------------------------------------------

x_min = float(X_test[x_feature].min())
x_max = float(X_test[x_feature].max())

y_min = float(X_test[y_feature].min())
y_max = float(X_test[y_feature].max())

if x_min == x_max:
    x_min -= 1
    x_max += 1

if y_min == y_max:
    y_min -= 1
    y_max += 1

x_padding = (x_max - x_min) * 0.05
y_padding = (y_max - y_min) * 0.05

x_range = [
    x_min - x_padding
    + i * ((x_max - x_min + 2 * x_padding) / 79)
    for i in range(80)
]

y_range = [
    y_min - y_padding
    + i * ((y_max - y_min + 2 * y_padding) / 79)
    for i in range(80)
]

# 2차원 격자 데이터를 1차원 행 목록으로 생성
grid_rows = []

for y_value in y_range:
    for x_value in x_range:
        row = {}

        for col in selected_features:
            if col == x_feature:
                row[col] = x_value
            elif col == y_feature:
                row[col] = y_value
            else:
                row[col] = fixed_values[col]

        grid_rows.append(row)

grid = pd.DataFrame(grid_rows, columns=selected_features)


# --------------------------------------------------
# 11. 로지스틱 회귀 결정 경계
# --------------------------------------------------

grid_scaled = (grid - train_mean) / train_std

logistic_grid_probability = logistic_model.predict_proba(
    grid_scaled
)[:, list(logistic_model.classes_).index(1)]

# 1차원 결과를 2차원 격자로 변환
logistic_probability_grid = []

for i in range(len(y_range)):
    start = i * len(x_range)
    end = start + len(x_range)

    logistic_probability_grid.append(
        logistic_grid_probability[start:end].tolist()
    )


# --------------------------------------------------
# 12. 의사결정트리 예측 영역
# --------------------------------------------------

tree_grid_prediction = tree_model.predict(grid)

tree_prediction_grid = []

for i in range(len(y_range)):
    start = i * len(x_range)
    end = start + len(x_range)

    tree_prediction_grid.append(
        tree_grid_prediction[start:end].tolist()
    )


# --------------------------------------------------
# 13. 테스트 데이터 산점도
# --------------------------------------------------

def add_test_points(fig):
    # 뇌졸중이 아닌 사람
    negative = test_df["stroke"] == 0

    fig.add_trace(
        go.Scatter(
            x=test_df.loc[negative, x_feature],
            y=test_df.loc[negative, y_feature],
            mode="markers",
            name="실제 뇌졸중 아님",
            marker=dict(
                color="#2878B5",
                size=6,
                opacity=0.65
            )
        )
    )

    # 뇌졸중인 사람
    positive = test_df["stroke"] == 1

    fig.add_trace(
        go.Scatter(
            x=test_df.loc[positive, x_feature],
            y=test_df.loc[positive, y_feature],
            mode="markers",
            name="실제 뇌졸중",
            marker=dict(
                color="#E45756",
                size=7,
                opacity=0.8
            )
        )
    )

    fig.update_layout(
        xaxis_title=FEATURE_NAMES[x_feature],
        yaxis_title=FEATURE_NAMES[y_feature],
        legend_title="실제 결과",
        height=550
    )

    return fig


# --------------------------------------------------
# 14. 로지스틱 회귀 그래프
# --------------------------------------------------

st.subheader("① 로지스틱 회귀 예측 영역")

logistic_fig = go.Figure()

# 확률에 따른 배경 영역
logistic_fig.add_trace(
    go.Heatmap(
        x=x_range,
        y=y_range,
        z=logistic_probability_grid,
        colorscale=[
            [0, "rgba(40,120,181,0.10)"],
            [0.5, "rgba(220,220,220,0.10)"],
            [1, "rgba(228,87,86,0.10)"]
        ],
        zmin=0,
        zmax=1,
        showscale=False,
        hovertemplate=(
            "x: %{x:.2f}<br>"
            "y: %{y:.2f}<br>"
            "뇌졸중 예측 확률: %{z:.3f}"
            "<extra></extra>"
        )
    )
)

# 확률 0.5 결정 경계
probability_min = min(logistic_grid_probability[0])
probability_max = max(
    max(row) for row in logistic_probability_grid
)

if probability_min <= 0.5 <= probability_max:
    logistic_fig.add_trace(
        go.Contour(
            x=x_range,
            y=y_range,
            z=logistic_probability_grid,
            contours=dict(
                start=0.5,
                end=0.5,
                size=0.01,
                coloring="none",
                showlabels=True
            ),
            line=dict(
                color="black",
                width=3
            ),
            showscale=False,
            name="확률 0.5 결정 경계",
            hoverinfo="skip"
        )
    )
else:
    st.info(
        "로지스틱 회귀의 확률 0.5 결정 경계가 "
        "현재 그래프 범위 안에 나타나지 않습니다."
    )

logistic_fig = add_test_points(logistic_fig)

st.plotly_chart(
    logistic_fig,
    use_container_width=True
)

st.caption(
    "검은 선은 뇌졸중 예측 확률이 0.5인 결정 경계입니다. "
    "배경은 모델이 계산한 예측 확률을 나타냅니다."
)


# --------------------------------------------------
# 15. 의사결정트리 예측 영역
# --------------------------------------------------

st.subheader("② 의사결정트리 예측 영역")

tree_fig = go.Figure()

tree_fig.add_trace(
    go.Heatmap(
        x=x_range,
        y=y_range,
        z=tree_prediction_grid,
        colorscale=[
            [0, "rgba(40,120,181,0.15)"],
            [0.499, "rgba(40,120,181,0.15)"],
            [0.5, "rgba(228,87,86,0.15)"],
            [1, "rgba(228,87,86,0.15)"]
        ],
        zmin=0,
        zmax=1,
        showscale=False,
        hovertemplate=(
            "x: %{x:.2f}<br>"
            "y: %{y:.2f}<br>"
            "예측 결과: %{z}"
            "<extra></extra>"
        )
    )
)

tree_fig = add_test_points(tree_fig)

st.plotly_chart(
    tree_fig,
    use_container_width=True
)

st.caption(
    "옅은 파란색 영역은 뇌졸중이 아니라고 예측한 영역이고, "
    "옅은 붉은색 영역은 뇌졸중이라고 예측한 영역입니다."
)


# --------------------------------------------------
# 16. 의사결정트리 질문 구조
# --------------------------------------------------

st.header("5. 의사결정트리 질문 구조")

st.write(
    "의사결정트리는 훈련용 데이터를 바탕으로 속성에 관한 질문을 "
    "최대 3회까지 나누어 예측합니다. 각 마지막 마디에는 최소 5명의 "
    "훈련용 데이터가 남도록 설정했습니다."
)

tree_structure = tree_model.tree_
tree_classes = list(tree_model.classes_)

positive_class_index = tree_classes.index(1)

leaf_count = 0
negative_leaf_count = 0
used_features = set()

dot_lines = [
    "digraph DecisionTree {",
    'graph [rankdir=TB, bgcolor="white"];',
    'node [shape=box, style="filled,rounded", fontname="sans-serif"];',
    'edge [fontname="sans-serif"];'
]


def make_node(node):
    global leaf_count, negative_leaf_count

    left_child = tree_structure.children_left[node]
    right_child = tree_structure.children_right[node]

    sample_count = int(tree_structure.n_node_samples[node])

    values = tree_structure.value[node][0].tolist()
    total_value = sum(values)

    if total_value > 0:
        positive_count = round(
            values[positive_class_index]
            / total_value
            * sample_count
        )
    else:
        positive_count = 0

    positive_rate = (
        positive_count / sample_count * 100
        if sample_count > 0 else 0
    )

    prediction_index = max(
        range(len(values)),
        key=lambda i: values[i]
    )

    prediction = int(tree_classes[prediction_index])

    # 마지막 마디
    if left_child == right_child:
        leaf_count += 1

        if prediction == 0:
            negative_leaf_count += 1
            fill_color = "#D7EAF7"
            answer = "뇌졸중 아님"
        else:
            fill_color = "#F8D9D9"
            answer = "뇌졸중"

        label = (
            f"마디 {node}\\n"
            f"훈련 인원: {sample_count}명\\n"
            f"뇌졸중: {positive_count}명\\n"
            f"뇌졸중 비율: {positive_rate:.1f}%\\n"
            f"예측: {answer}"
        )

        dot_lines.append(
            f'n{node} [label="{label}", fillcolor="{fill_color}"];'
        )

        return

    # 질문이 있는 내부 마디
    feature_index = int(tree_structure.feature[node])
    feature_name = selected_features[feature_index]
    feature_korean = FEATURE_NAMES[feature_name]

    threshold = float(tree_structure.threshold[node])

    used_features.add(feature_name)

    label = (
        f"마디 {node}\\n"
        f"훈련 인원: {sample_count}명\\n"
        f"뇌졸중: {positive_count}명\\n"
        f"뇌졸중 비율: {positive_rate:.1f}%\\n"
        f"질문: {feature_korean} ≤ {threshold:.2f}?"
    )

    dot_lines.append(
        f'n{node} [label="{label}", fillcolor="#FFF0C2"];'
    )

    dot_lines.append(
        f'n{node} -> n{left_child} [label="예"];'
    )

    dot_lines.append(
        f'n{node} -> n{right_child} [label="아니요"];'
    )

    make_node(left_child)
    make_node(right_child)


make_node(0)

dot_lines.append("}")

dot_source = "\n".join(dot_lines)

st.graphviz_chart(
    dot_source,
    use_container_width=True
)


# --------------------------------------------------
# 17. 의사결정트리 요약
# --------------------------------------------------

st.subheader("의사결정트리 구조 요약")

st.write(f"- 예측을 내리는 마지막 마디 수: **{leaf_count}개**")
st.write(
    f"- 뇌졸중이 아니라고 예측하는 마지막 마디 수: "
    f"**{negative_leaf_count}개**"
)

if used_features:
    used_feature_text = ", ".join(
        FEATURE_NAMES[col] for col in selected_features
        if col in used_features
    )
else:
    used_feature_text = "질문에 사용된 속성이 없습니다."

st.write(f"- 실제 질문에 사용된 속성: **{used_feature_text}**")

st.write(
    "각 마디의 훈련 인원과 뇌졸중 인원은 "
    "의사결정트리가 학습할 때 사용한 훈련용 데이터 기준입니다."
)


# --------------------------------------------------
# 18. 최종 성능 비교
# --------------------------------------------------

st.header("6. 최종 성능 비교")

comparison_table = pd.DataFrame({
    "모델": [
        "로지스틱 회귀",
        "의사결정트리",
        "기준 모델"
    ],
    "훈련용 정확도(%)": [
        round(logistic_train_accuracy * 100, 2),
        round(tree_train_accuracy * 100, 2),
        round(baseline_train_accuracy * 100, 2)
    ],
    "테스트용 정확도(%)": [
        round(logistic_test_accuracy * 100, 2),
        round(tree_test_accuracy * 100, 2),
        round(baseline_test_accuracy * 100, 2)
    ]
})

st.dataframe(
    comparison_table,
    use_container_width=True
)

st.write(
    "정확도는 전체 예측 중 실제 결과와 일치한 예측의 비율입니다. "
    "뇌졸중 데이터처럼 두 결과의 인원 차이가 큰 자료에서는 "
    "정확도만으로 모델의 성능을 판단하기 어렵습니다."
)

st.warning(
    "이 실습은 분류 알고리즘의 원리와 성능 비교를 학습하기 위한 "
    "교육용 활동입니다. 실제 환자의 뇌졸중 발생 여부를 진단하거나 "
    "의료적 판단을 내리는 용도로 사용할 수 없습니다."
)
