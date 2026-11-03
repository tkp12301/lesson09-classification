
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ==================================================
# 1. 기본 설정
# ==================================================

st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 뇌졸중 예측 실습실")
st.write("데이터 전처리부터 분류 모델 학습과 성능 비교까지 진행합니다.")

DATA_URL = "https://raw.githubusercontent.com/tkp12301/data/refs/heads/main/stroke.csv"

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}


# ==================================================
# 2. 데이터 불러오기
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


# ==================================================
# 3. 분석 속성 선택
# ==================================================

st.header("1. 분석할 속성 선택")

selected_features = st.multiselect(
    "뇌졸중 예측에 사용할 속성을 선택하세요.",
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
    st.warning("속성을 최소 2개 이상 선택해야 합니다.")
    st.stop()

st.write(
    "선택한 속성: "
    + ", ".join(FEATURE_NAMES[x] for x in selected_features)
)


# ==================================================
# 4. 데이터 기본 정리
# ==================================================

st.header("2. 데이터 전처리")

st.subheader("① 데이터 기본 확인")

required_columns = ["id", "stroke"] + selected_features

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        "필요한 데이터 열이 없습니다: "
        + ", ".join(missing_columns)
    )
    st.stop()

st.write(f"- 원본 데이터 행 수: **{len(df):,}명**")
st.write(f"- 원본 데이터 열 수: **{len(df.columns)}개**")

df["id"] = pd.to_numeric(df["id"], errors="coerce")
df["stroke"] = pd.to_numeric(df["stroke"], errors="coerce")

df = df.dropna(subset=["id", "stroke"]).copy()
df = df[df["stroke"].isin([0, 1])].copy()

df["stroke"] = df["stroke"].astype(int)

for col in selected_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values(
    by="id",
    kind="mergesort"
).reset_index(drop=True)

missing_table = pd.DataFrame({
    "속성": [FEATURE_NAMES[col] for col in selected_features],
    "결측값 수": [
        int(df[col].isna().sum())
        for col in selected_features
    ]
})

st.write("**선택한 속성의 결측값 현황**")
st.dataframe(missing_table, use_container_width=True)


# ==================================================
# 5. 훈련용 / 테스트용 데이터 분리
# ==================================================

st.subheader("② 훈련용 데이터와 테스트용 데이터 분리")

st.write(
    "ID 번호 순서로 정렬한 후 10명씩 묶어 "
    "앞의 3명은 테스트용, 나머지 7명은 훈련용으로 분리합니다."
)

test_mask = [
    i % 10 < 3
    for i in range(len(df))
]

train_mask = [
    not value
    for value in test_mask
]

train_df = df.loc[train_mask].copy()
test_df = df.loc[test_mask].copy()

st.write(f"- 전체 데이터: **{len(df):,}명**")
st.write(f"- 훈련용 데이터: **{len(train_df):,}명**")
st.write(f"- 테스트용 데이터: **{len(test_df):,}명**")

if len(train_df) == 0 or len(test_df) == 0:
    st.error("훈련용 또는 테스트용 데이터가 없습니다.")
    st.stop()

if train_df["stroke"].nunique() != 2:
    st.error("훈련용 데이터에 뇌졸중 여부 두 종류가 모두 있어야 합니다.")
    st.stop()

split_table = pd.DataFrame({
    "구분": ["훈련용 데이터", "테스트용 데이터"],
    "전체 인원": [len(train_df), len(test_df)],
    "뇌졸중 인원": [
        int(train_df["stroke"].sum()),
        int(test_df["stroke"].sum())
    ],
    "뇌졸중 비율(%)": [
        round(train_df["stroke"].mean() * 100, 2),
        round(test_df["stroke"].mean() * 100, 2)
    ]
})

st.dataframe(split_table, use_container_width=True)


# ==================================================
# 6. 결측값 처리
# ==================================================

st.subheader("③ 결측값 처리")

st.write(
    "BMI를 선택한 경우 훈련용 데이터의 중앙값을 계산하여 "
    "훈련용과 테스트용 BMI의 결측값을 채웁니다. "
    "다른 선택 속성에 결측값이 있다면 동일하게 훈련용 중앙값을 사용합니다."
)

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].copy()
y_test = test_df["stroke"].copy()

preprocessing_records = []

for col in selected_features:
    train_missing = int(X_train[col].isna().sum())
    test_missing = int(X_test[col].isna().sum())

    if train_missing > 0 or test_missing > 0:
        median_value = X_train[col].median()

        if pd.isna(median_value):
            st.error(
                f"{FEATURE_NAMES[col]} 속성의 훈련용 데이터가 모두 결측값입니다."
            )
            st.stop()

        X_train.loc[:, col] = X_train[col].fillna(median_value)
        X_test.loc[:, col] = X_test[col].fillna(median_value)

        preprocessing_records.append({
            "속성": FEATURE_NAMES[col],
            "처리 방법": "훈련용 데이터 중앙값으로 대체",
            "중앙값": round(float(median_value), 3),
            "훈련용 처리 인원": train_missing,
            "테스트용 처리 인원": test_missing
        })

if preprocessing_records:
    st.dataframe(
        pd.DataFrame(preprocessing_records),
        use_container_width=True
    )
else:
    st.info("선택한 속성에 결측값이 없습니다.")

if X_train.isna().any().any() or X_test.isna().any().any():
    st.error("결측값 처리가 완료되지 않았습니다.")
    st.stop()

st.write(
    f"- 전처리 후 훈련용 결측값: **{int(X_train.isna().sum().sum())}개**"
)
st.write(
    f"- 전처리 후 테스트용 결측값: **{int(X_test.isna().sum().sum())}개**"
)


# ==================================================
# 7. 로지스틱 회귀 표준화
# ==================================================

st.subheader("④ 데이터 크기 맞추기(표준화)")

st.write(
    "로지스틱 회귀는 훈련용 데이터의 평균과 표준편차를 이용해 "
    "속성별 데이터 크기를 맞춥니다. 테스트용 데이터에도 "
    "훈련용 데이터에서 계산한 동일한 기준을 적용합니다."
)

train_mean = X_train.mean()
train_std = X_train.std()

train_std = train_std.fillna(1)
train_std = train_std.replace(0, 1)

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

st.write("**표준화된 훈련용 데이터 미리보기**")
st.dataframe(X_train_scaled.head(), use_container_width=True)


# ==================================================
# 8. 분류 모델 학습
# ==================================================

st.header("3. 분류 모델 학습")

st.write(
    "전처리가 완료된 훈련용 데이터만 사용하여 "
    "로지스틱 회귀와 의사결정트리를 학습합니다."
)

# 로지스틱 회귀 모델
logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

logistic_model.fit(X_train_scaled, y_train)

# 의사결정트리 모델
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(X_train, y_train)

st.subheader("① 모델 학습 과정")

training_table = pd.DataFrame({
    "모델": [
        "로지스틱 회귀(확률로 답하는 모델)",
        "의사결정트리(질문으로 답하는 모델)"
    ],
    "학습 인원": [
        len(X_train),
        len(X_train)
    ],
    "사용 속성 수": [
        len(selected_features),
        len(selected_features)
    ],
    "학습 데이터": [
        "표준화된 훈련용 데이터",
        "전처리된 훈련용 데이터"
    ]
})

st.dataframe(training_table, use_container_width=True)

st.write("**모델 학습 설정**")
st.write("- 로지스틱 회귀: 최대 반복 횟수 2,000회")
st.write("- 의사결정트리: 최대 깊이 3, 마지막 마디 최소 인원 5명")
st.write("- 두 모델 모두 난수 고정값 42 사용")


# ==================================================
# 9. 모델 예측 및 정확도
# ==================================================

st.subheader("② 모델 성능 평가")

logistic_train_pred = logistic_model.predict(X_train_scaled)
logistic_test_pred = logistic_model.predict(X_test_scaled)

tree_train_pred = tree_model.predict(X_train)
tree_test_pred = tree_model.predict(X_test)

# 기준 모델: 훈련용 데이터에서 다수인 결과만 예측
majority_class = int(y_train.value_counts().idxmax())

baseline_train_pred = [majority_class] * len(y_train)
baseline_test_pred = [majority_class] * len(y_test)


def calculate_accuracy(actual, predicted):
    actual_values = actual.reset_index(drop=True)
    predicted_values = pd.Series(predicted).reset_index(drop=True)

    return float((actual_values == predicted_values).mean())


logistic_train_accuracy = calculate_accuracy(
    y_train, logistic_train_pred
)
logistic_test_accuracy = calculate_accuracy(
    y_test, logistic_test_pred
)

tree_train_accuracy = calculate_accuracy(
    y_train, tree_train_pred
)
tree_test_accuracy = calculate_accuracy(
    y_test, tree_test_pred
)

baseline_train_accuracy = calculate_accuracy(
    y_train, baseline_train_pred
)
baseline_test_accuracy = calculate_accuracy(
    y_test, baseline_test_pred
)


# 정확도 카드
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "로지스틱 회귀",
        f"{logistic_test_accuracy * 100:.2f}%"
    )
    st.caption(f"훈련용 정확도: {logistic_train_accuracy * 100:.2f}%")
    st.caption(f"테스트용 정확도: {logistic_test_accuracy * 100:.2f}%")

with col2:
    st.metric(
        "의사결정트리",
        f"{tree_test_accuracy * 100:.2f}%"
    )
    st.caption(f"훈련용 정확도: {tree_train_accuracy * 100:.2f}%")
    st.caption(f"테스트용 정확도: {tree_test_accuracy * 100:.2f}%")

with col3:
    st.metric(
        "기준 모델",
        f"{baseline_test_accuracy * 100:.2f}%"
    )
    st.caption(f"훈련용 정확도: {baseline_train_accuracy * 100:.2f}%")
    st.caption(f"테스트용 정확도: {baseline_test_accuracy * 100:.2f}%")

if majority_class == 1:
    st.write("기준 모델은 모든 사람을 뇌졸중이라고 예측합니다.")
else:
    st.write("기준 모델은 모든 사람을 뇌졸중이 아니라고 예측합니다.")


# ==================================================
# 10. 그래프 축 선택
# ==================================================

st.header("4. 분류 결과 시각화")

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "가로축 속성",
        selected_features,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x]
    )

y_options = [
    col for col in selected_features
    if col != x_feature
]

with axis_col2:
    y_feature = st.selectbox(
        "세로축 속성",
        y_options,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x]
    )

# 그래프 축 이외의 속성은 테스트 데이터 중앙값으로 고정
fixed_values = {}

for col in selected_features:
    if col not in [x_feature, y_feature]:
        fixed_values[col] = float(X_test[col].median())

if fixed_values:
    fixed_text = ", ".join([
        f"{FEATURE_NAMES[col]} = {value:.2f}"
        for col, value in fixed_values.items()
    ])

    st.info(
        "그래프 축 이외의 속성은 테스트 데이터 중앙값으로 고정했습니다: "
        + fixed_text
    )


# ==================================================
# 11. 그래프용 격자 생성
# ==================================================

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
    + i * (x_max - x_min + 2 * x_padding) / 79
    for i in range(80)
]

y_range = [
    y_min - y_padding
    + i * (y_max - y_min + 2 * y_padding) / 79
    for i in range(80)
]

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

grid = pd.DataFrame(
    grid_rows,
    columns=selected_features
)


# ==================================================
# 12. 로지스틱 회귀 확률 계산
# ==================================================

st.subheader("① 로지스틱 회귀 예측 영역")

# 격자 데이터도 훈련용 평균과 표준편차로 표준화
grid_scaled = (grid - train_mean) / train_std

# 뇌졸중(1)에 해당하는 확률 열의 위치 확인
positive_index = list(logistic_model.classes_).index(1)

# 모든 격자 지점에서 뇌졸중일 확률 계산
probability_array = logistic_model.predict_proba(grid_scaled)

# 확률 결과를 일반 파이썬 리스트로 변환
probability_values = probability_array[:, positive_index].tolist()

# 80개씩 끊어 80행의 2차원 리스트 생성
probability_grid = []

for row_number in range(len(y_range)):
    start_index = row_number * len(x_range)
    end_index = start_index + len(x_range)

    probability_grid.append(
        probability_values[start_index:end_index]
    )

# 전체 확률의 최솟값과 최댓값 계산
all_probabilities = [
    value
    for row in probability_grid
    for value in row
]

probability_min = min(all_probabilities)
probability_max = max(all_probabilities)

logistic_fig = go.Figure()

# 확률 배경
logistic_fig.add_trace(
    go.Heatmap(
        x=x_range,
        y=y_range,
        z=probability_grid,
        colorscale=[
            [0.0, "rgba(40,120,181,0.12)"],
            [0.5, "rgba(220,220,220,0.12)"],
            [1.0, "rgba(228,87,86,0.12)"]
        ],
        zmin=0,
        zmax=1,
        showscale=False,
        hovertemplate=(
            "가로축: %{x:.2f}<br>"
            "세로축: %{y:.2f}<br>"
            "뇌졸중 확률: %{z:.3f}"
            "<extra></extra>"
        )
    )
)

# 확률 0.5 결정 경계
if probability_min <= 0.5 <= probability_max:
    logistic_fig.add_trace(
        go.Contour(
            x=x_range,
            y=y_range,
            z=probability_grid,
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
        "확률 0.5 결정 경계가 현재 그래프 범위 안에 없습니다. "
        "모델의 확률이 그래프 전체에서 0.5보다 크거나 작습니다."
    )


# ==================================================
# 13. 테스트 데이터 점 추가
# ==================================================

def add_test_scatter(fig):
    negative_mask = test_df["stroke"] == 0
    positive_mask = test_df["stroke"] == 1

    fig.add_trace(
        go.Scatter(
            x=test_df.loc[negative_mask, x_feature],
            y=test_df.loc[negative_mask, y_feature],
            mode="markers",
            name="실제 뇌졸중 아님",
            marker=dict(
                color="#2878B5",
                size=6,
                opacity=0.65
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=test_df.loc[positive_mask, x_feature],
            y=test_df.loc[positive_mask, y_feature],
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


logistic_fig = add_test_scatter(logistic_fig)

st.plotly_chart(
    logistic_fig,
    use_container_width=True
)

st.caption(
    "검은 선은 뇌졸중 예측 확률이 0.5인 결정 경계입니다. "
    "파란색과 붉은색 점은 테스트 데이터의 실제 결과입니다."
)


# ==================================================
# 14. 의사결정트리 예측 영역
# ==================================================

st.subheader("② 의사결정트리 예측 영역")

tree_grid_predictions = tree_model.predict(grid).tolist()

tree_prediction_grid = []

for row_number in range(len(y_range)):
    start_index = row_number * len(x_range)
    end_index = start_index + len(x_range)

    tree_prediction_grid.append(
        tree_grid_predictions[start_index:end_index]
    )

tree_fig = go.Figure()

tree_fig.add_trace(
    go.Heatmap(
        x=x_range,
        y=y_range,
        z=tree_prediction_grid,
        colorscale=[
            [0.0, "rgba(40,120,181,0.15)"],
            [0.499, "rgba(40,120,181,0.15)"],
            [0.5, "rgba(228,87,86,0.15)"],
            [1.0, "rgba(228,87,86,0.15)"]
        ],
        zmin=0,
        zmax=1,
        showscale=False,
        hovertemplate=(
            "가로축: %{x:.2f}<br>"
            "세로축: %{y:.2f}<br>"
            "예측 결과: %{z}"
            "<extra></extra>"
        )
    )
)

tree_fig = add_test_scatter(tree_fig)

st.plotly_chart(
    tree_fig,
    use_container_width=True
)

st.caption(
    "파란색 영역은 뇌졸중이 아니라고 예측한 영역이며, "
    "붉은색 영역은 뇌졸중이라고 예측한 영역입니다."
)


# ==================================================
# 15. 의사결정트리 질문 구조
# ==================================================

st.header("5. 의사결정트리 질문 구조")

st.write(
    "의사결정트리는 최대 3번까지 질문하며, "
    "마지막 마디에는 최소 5명의 훈련용 데이터가 남도록 설정했습니다."
)

tree_structure = tree_model.tree_
tree_classes = list(tree_model.classes_)
positive_class_index = tree_classes.index(1)

tree_summary = {
    "leaf_count": 0,
    "negative_leaf_count": 0,
    "used_features": set()
}

dot_lines = [
    "digraph DecisionTree {",
    'graph [rankdir=TB, bgcolor="white"];',
    'node [shape=box, style="filled,rounded", fontname="sans-serif"];',
    'edge [fontname="sans-serif"];'
]


def make_node(node):
    left_child = tree_structure.children_left[node]
    right_child = tree_structure.children_right[node]

    sample_count = int(tree_structure.n_node_samples[node])

    values = tree_structure.value[node][0].tolist()
    total_value = sum(values)

    if total_value > 0:
        positive_count = round(
            values[positive_class_index] / total_value * sample_count
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

    # 리프 노드
    if left_child == right_child:
        tree_summary["leaf_count"] += 1

        if prediction == 0:
            tree_summary["negative_leaf_count"] += 1
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

    # 질문 노드
    feature_index = int(tree_structure.feature[node])
    feature_name = selected_features[feature_index]
    feature_korean = FEATURE_NAMES[feature_name]
    threshold = float(tree_structure.threshold[node])

    tree_summary["used_features"].add(feature_name)

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


# ==================================================
# 16. 의사결정트리 요약
# ==================================================

st.subheader("의사결정트리 요약")

st.write(
    f"- 예측을 내리는 마지막 마디 수: "
    f"**{tree_summary['leaf_count']}개**"
)

st.write(
    f"- 뇌졸중이 아니라고 예측하는 마지막 마디 수: "
    f"**{tree_summary['negative_leaf_count']}개**"
)

used_feature_names = [
    FEATURE_NAMES[col]
    for col in selected_features
    if col in tree_summary["used_features"]
]

if used_feature_names:
    st.write(
        "- 실제 질문에 사용된 속성: **"
        + ", ".join(used_feature_names)
        + "**"
    )
else:
    st.write("- 실제 질문에 사용된 속성이 없습니다.")


# ==================================================
# 17. 최종 비교
# ==================================================

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
    "뇌졸중 데이터처럼 뇌졸중인 사람과 아닌 사람의 수에 차이가 큰 경우 "
    "정확도만으로 모델의 성능을 판단하기 어렵습니다."
)

st.warning(
    "이 프로그램은 분류 알고리즘을 학습하기 위한 교육용 실습입니다. "
    "실제 환자의 뇌졸중 여부를 진단하거나 의료적 판단을 내리는 용도로 "
    "사용할 수 없습니다."
)
