
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ==================================================
# 1. 페이지 설정
# ==================================================

st.set_page_config(
    page_title="뇌졸중 분류 모델",
    page_icon="🤖",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/tkp12301/data/refs/heads/main/stroke.csv"

RANDOM_STATE = 42

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

FEATURES = list(FEATURE_NAMES.keys())


# ==================================================
# 2. 데이터 불러오기
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🤖 뇌졸중 분류 모델")
st.write(
    "선택한 건강 정보를 바탕으로 뇌졸중 여부를 분류하는 "
    "머신러닝 모델을 만들고 성능을 비교합니다."
)

try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(e))
    st.stop()


# 숫자형 변환
for col in FEATURES + ["stroke", "id"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ==================================================
# 3. 입력 속성 선택
# ==================================================

st.divider()
st.header("1. 모델에 사용할 속성 선택")

st.write(
    "뇌졸중 여부를 예측하는 데 사용할 속성을 선택하세요. "
    "처음에는 체질량지수를 제외한 네 가지 속성이 선택되어 있습니다."
)

selected_features = st.multiselect(
    "입력 속성",
    options=FEATURES,
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease"
    ],
    format_func=lambda x: FEATURE_NAMES[x]
)

if len(selected_features) < 2:
    st.warning("입력 속성을 두 개 이상 선택해 주세요.")
    st.stop()


# ==================================================
# 4. 데이터 정렬 및 학습 / 테스트 분리
# ==================================================

st.divider()
st.header("2. 학습 데이터와 테스트 데이터 분리")

# ID 번호 순서로 정렬
df = df.sort_values(
    by="id",
    ascending=True,
    kind="mergesort"
).reset_index(drop=True)

# 정답이 확인된 데이터만 사용
df = df[df["stroke"].isin([0, 1])].reset_index(drop=True)

# 10명씩 묶었을 때 앞의 3명은 테스트 데이터
test_mask = (np.arange(len(df)) % 10) < 3

train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()

st.write(
    "ID 번호 순으로 정렬한 뒤 10명씩 묶고, "
    "각 묶음의 앞 3명을 테스트 데이터로 분리했습니다."
)

col1, col2 = st.columns(2)

col1.metric(
    "훈련 데이터",
    f"{len(train_df):,}명"
)

col2.metric(
    "테스트 데이터",
    f"{len(test_df):,}명"
)

if len(test_df) != 1533:
    st.warning(
        f"테스트 데이터가 요청한 1,533명과 다릅니다. "
        f"현재 데이터의 테스트 인원은 {len(test_df):,}명입니다."
    )

if len(train_df) == 0 or len(test_df) == 0:
    st.error("훈련 데이터 또는 테스트 데이터가 없습니다.")
    st.stop()


# ==================================================
# 5. 결측값 처리
# ==================================================

st.divider()
st.header("3. 데이터 전처리")

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

# BMI를 선택한 경우에만 훈련 데이터 중앙값으로 채움
if "bmi" in selected_features:

    bmi_median = X_train["bmi"].median()

    if pd.isna(bmi_median):
        st.error("훈련 데이터에서 BMI 중앙값을 계산할 수 없습니다.")
        st.stop()

    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

    st.write(
        f"체질량지수의 빈 값은 훈련 데이터의 중앙값 "
        f"{bmi_median:.2f}으로 채웠습니다."
    )

# 선택한 다른 속성에 결측값이 있으면 별도로 안내
if X_train.isna().any().any() or X_test.isna().any().any():
    st.error(
        "선택한 속성 중 BMI 이외의 빈 값이 발견되었습니다. "
        "요청한 전처리 조건에 따라 임의로 채우지 않고 학습을 중단합니다."
    )
    st.stop()

# 두 클래스가 모두 존재하는지 확인
if y_train.nunique() < 2:
    st.error("훈련 데이터에 뇌졸중 여부의 두 클래스가 모두 있어야 합니다.")
    st.stop()


# ==================================================
# 6. 모델 생성
# ==================================================

st.divider()
st.header("4. 분류 모델 학습")

# 로지스틱 회귀용 크기 맞추기
# 훈련 데이터로만 평균과 표준편차 계산
train_mean = X_train.mean()
train_std = X_train.std().replace(0, 1)

X_train_scaled = (X_train - train_mean) / train_std
X_test_scaled = (X_test - train_mean) / train_std

# 로지스틱 회귀
logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

# 의사결정트리
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE
)

tree_model.fit(
    X_train,
    y_train
)

# 의사결정트리는 변수의 크기 단위에 영향을 받지 않으므로
# 원래 단위의 훈련 데이터로 학습함.


# ==================================================
# 7. 정확도 계산
# ==================================================

# 로지스틱 회귀
logistic_train_accuracy = logistic_model.score(
    X_train_scaled,
    y_train
)

logistic_test_accuracy = logistic_model.score(
    X_test_scaled,
    y_test
)

# 의사결정트리
tree_train_accuracy = tree_model.score(
    X_train,
    y_train
)

tree_test_accuracy = tree_model.score(
    X_test,
    y_test
)

# 다수 클래스 기준 모델
majority_class = int(y_train.value_counts().idxmax())

baseline_train_accuracy = np.mean(
    y_train == majority_class
)

baseline_test_accuracy = np.mean(
    y_test == majority_class
)


# ==================================================
# 8. 정확도 카드
# ==================================================

st.divider()
st.header("5. 모델 성능 비교")

st.caption(
    "큰 숫자는 테스트 데이터 정확도이며, "
    "아래에는 훈련 데이터와 테스트 데이터 정확도를 함께 표시합니다."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "로지스틱 회귀(확률로 답하는 모델)",
        f"{logistic_test_accuracy * 100:.2f}%"
    )

    st.caption(
        f"훈련 데이터: {logistic_train_accuracy * 100:.2f}%"
    )

    st.caption(
        f"테스트 데이터: {logistic_test_accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "의사결정트리(질문으로 답하는 모델)",
        f"{tree_test_accuracy * 100:.2f}%"
    )

    st.caption(
        f"훈련 데이터: {tree_train_accuracy * 100:.2f}%"
    )

    st.caption(
        f"테스트 데이터: {tree_test_accuracy * 100:.2f}%"
    )

with col3:
    st.metric(
        "다수 클래스 기준 모델",
        f"{baseline_test_accuracy * 100:.2f}%"
    )

    st.caption(
        f"훈련 데이터: {baseline_train_accuracy * 100:.2f}%"
    )

    st.caption(
        f"테스트 데이터: {baseline_test_accuracy * 100:.2f}%"
    )

st.info(
    f"다수 클래스 기준 모델은 훈련 데이터에서 더 많은 정답인 "
    f"{'뇌졸중' if majority_class == 1 else '뇌졸중 아님'}만 "
    "계속 예측하는 모델입니다."
)


# ==================================================
# 9. 2차원 분류 그림
# ==================================================

st.divider()
st.header("6. 두 속성으로 분류 결과 시각화")

st.write(
    "선택한 속성 중 두 가지를 골라 가로축과 세로축으로 설정합니다. "
    "점의 색은 실제 뇌졸중 여부를 나타냅니다."
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
    y_default = 1 if len(selected_features) > 1 else 0

    y_feature = st.selectbox(
        "세로축 속성",
        options=selected_features,
        index=y_default,
        format_func=lambda x: FEATURE_NAMES[x]
    )

if x_feature == y_feature:
    st.warning("가로축과 세로축은 서로 다른 속성을 선택해야 합니다.")
    st.stop()


# 테스트 데이터의 중앙값으로 나머지 속성 고정
fixed_values = {}

for feature in selected_features:

    if feature not in [x_feature, y_feature]:

        fixed_values[feature] = X_test[feature].median()

st.write("**그림을 그릴 때 고정한 나머지 속성의 값**")

if fixed_values:

    fixed_table = pd.DataFrame([
        {
            "속성": FEATURE_NAMES[key],
            "고정한 값": round(float(value), 3)
        }
        for key, value in fixed_values.items()
    ])

    st.dataframe(
        fixed_table,
        use_container_width=True,
        hide_index=True
    )

else:
    st.write("선택한 속성이 두 개뿐이므로 고정할 나머지 속성이 없습니다.")


# 축 범위 설정
x_min = float(X_test[x_feature].min())
x_max = float(X_test[x_feature].max())

y_min = float(X_test[y_feature].min())
y_max = float(X_test[y_feature].max())

x_padding = max((x_max - x_min) * 0.08, 0.1)
y_padding = max((y_max - y_min) * 0.08, 0.1)

x_range = np.linspace(
    x_min - x_padding,
    x_max + x_padding,
    100
)

y_range = np.linspace(
    y_min - y_padding,
    y_max + y_padding,
    100
)

xx, yy = np.meshgrid(x_range, y_range)

grid = pd.DataFrame(
    np.tile(
        X_test.median().values,
        (xx.size, 1)
    ),
    columns=selected_features
)

grid[x_feature] = xx.ravel()
grid[y_feature] = yy.ravel()


# 로지스틱 회귀의 확률 계산
grid_scaled = (grid - train_mean) / train_std

grid_probability = logistic_model.predict_proba(
    grid_scaled
)[:, 1]

probability_grid = grid_probability.reshape(xx.shape)


# 의사결정트리 분류 영역 계산
tree_grid = tree_model.predict(grid)

tree_grid = tree_grid.reshape(xx.shape)


# 실제 테스트 데이터
fig = go.Figure()

# 의사결정트리의 분류 영역을 옅은 색으로 표시
fig.add_trace(
    go.Heatmap(
        x=x_range,
        y=y_range,
        z=tree_grid,
        colorscale=[
            [0, "rgba(100,149,237,0.18)"],
            [1, "rgba(255,99,71,0.18)"]
        ],
        showscale=False,
        hoverinfo="skip",
        name="의사결정트리 영역"
    )
)

# 로지스틱 회귀의 0.5 확률 경계선
fig.add_trace(
    go.Contour(
        x=x_range,
        y=y_range,
        z=probability_grid,
        contours=dict(
            start=0.5,
            end=0.5,
            size=1,
            coloring="none",
            showlabels=True
        ),
        line=dict(
            color="black",
            width=3
        ),
        showscale=False,
        name="로지스틱 회귀 0.5 경계선"
    )
)

# 실제 뇌졸중이 아닌 사람
negative = test_df["stroke"] == 0

fig.add_trace(
    go.Scatter(
        x=test_df.loc[negative, x_feature],
        y=test_df.loc[negative, y_feature],
        mode="markers",
        name="실제 뇌졸중 아님",
        marker=dict(
            color="royalblue",
            size=6,
            opacity=0.65
        )
    )
)

# 실제 뇌졸중인 사람
positive = test_df["stroke"] == 1

fig.add_trace(
    go.Scatter(
        x=test_df.loc[positive, x_feature],
        y=test_df.loc[positive, y_feature],
        mode="markers",
        name="실제 뇌졸중",
        marker=dict(
            color="red",
            size=7,
            opacity=0.8
        )
    )
)

fig.update_layout(
    title="실제 테스트 데이터와 모델의 분류 영역",
    xaxis_title=FEATURE_NAMES[x_feature],
    yaxis_title=FEATURE_NAMES[y_feature],
    legend_title="실제 뇌졸중 여부",
    height=650
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# 0.5 경계선이 그림 안에 있는지 확인
prob_min = float(probability_grid.min())
prob_max = float(probability_grid.max())

if prob_min <= 0.5 <= prob_max:

    st.success(
        "로지스틱 회귀의 0.5 확률 경계선이 그림의 범위 안에 있습니다."
    )

else:

    st.warning(
        "로지스틱 회귀의 0.5 확률 경계선이 그림의 범위 밖에 있습니다. "
        "현재 그림 범위에서는 경계선을 확인할 수 없습니다."
    )


# ==================================================
# 10. 의사결정트리 가지 그림
# ==================================================

st.divider()
st.header("7. 의사결정트리의 질문 구조")

st.write(
    "의사결정트리는 최대 세 번까지 질문하며, "
    "마지막 마디에 다섯 명 미만이 남으면 더 나누지 않습니다."
)

tree = tree_model.tree_

dot_lines = [
    "digraph DecisionTree {",
    'graph [rankdir=TB, bgcolor="white", nodesep=0.35, ranksep=0.55];',
    'node [shape=box, style="rounded,filled", fontname="Malgun Gothic", fontsize=10];',
    'edge [fontname="Malgun Gothic", fontsize=10];'
]

leaf_count = 0
negative_leaf_count = 0
used_features = set()


def make_node(node_id):

    global leaf_count, negative_leaf_count

    samples = int(tree.n_node_samples[node_id])

    values = tree.value[node_id][0]
    total_value = float(np.sum(values))

    if total_value > 0:
        positive_ratio = float(values[1] / total_value)
    else:
        positive_ratio = 0.0

    positive_count = int(round(positive_ratio * samples))

    positive_percent = (
        positive_count / samples * 100
        if samples > 0 else 0
    )

    is_leaf = tree.children_left[node_id] == tree.children_right[node_id]

    if is_leaf:

        leaf_count += 1

        prediction = int(tree.classes_[np.argmax(values)])

        if prediction == 0:
            negative_leaf_count += 1
            fill_color = "#B8D8F5"
            answer = "뇌졸중 아님"
        else:
            fill_color = "#F5B7B1"
            answer = "뇌졸중"

        label = (
            f"마디 {node_id}\\n"
            f"훈련 인원: {samples}명\\n"
            f"실제 뇌졸중: {positive_count}명\\n"
            f"비율: {positive_percent:.1f}%\\n"
            f"답: {answer}"
        )

        dot_lines.append(
            f'{node_id} [label="{label}", fillcolor="{fill_color}"];'
        )

        return

    feature_index = tree.feature[node_id]
    feature_name = selected_features[feature_index]
    threshold = tree.threshold[node_id]

    used_features.add(feature_name)

    label = (
        f"마디 {node_id}\\n"
        f"{FEATURE_NAMES[feature_name]} ≤ {threshold:.3f}?\\n"
        f"훈련 인원: {samples}명\\n"
        f"실제 뇌졸중: {positive_count}명\\n"
        f"비율: {positive_percent:.1f}%"
    )

    dot_lines.append(
        f'{node_id} [label="{label}", fillcolor="#FFF2CC"];'
    )

    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    make_node(left_child)
    make_node(right_child)

    dot_lines.append(
        f'{node_id} -> {left_child} [label="예"];'
    )

    dot_lines.append(
        f'{node_id} -> {right_child} [label="아니요"];'
    )


make_node(0)

dot_lines.append("}")

dot_string = "\n".join(dot_lines)

st.graphviz_chart(
    dot_string,
    use_container_width=True
)


# ==================================================
# 11. 의사결정트리 요약
# ==================================================

st.subheader("의사결정트리 구조 요약")

st.write(
    f"① 답을 내리는 마디는 총 **{leaf_count}개**이며, "
    f"그중 뇌졸중이 아니라고 답하는 마디는 "
    f"**{negative_leaf_count}개**입니다."
)

used_names = [
    FEATURE_NAMES[name]
    for name in selected_features
    if name in used_features
]

if used_names:

    st.write(
        "② 의사결정트리가 실제 질문에 사용한 속성: "
        + ", ".join(used_names)
    )

else:

    st.write(
        "② 의사결정트리가 실제 질문에 사용한 속성이 없습니다. "
        "훈련 데이터의 다수 클래스로만 분류했습니다."
    )


# ==================================================
# 12. 최종 안내
# ==================================================

st.divider()
st.subheader("모델 해석 시 주의사항")

st.info(
    "정확도는 전체 예측 중 정답을 맞힌 비율입니다. "
    "뇌졸중처럼 양성과 음성의 수가 크게 다른 데이터에서는 "
    "정확도만으로 모델의 성능을 판단하기 어렵습니다. "
    "특히 뇌졸중을 놓치지 않는 능력을 확인하려면 "
    "재현율과 혼동행렬도 함께 살펴볼 필요가 있습니다. "
    "이 실습 모델은 교육용이며 실제 의료 진단에 사용할 수 없습니다."
)
