
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 1. 페이지 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="성능 높이기",
    page_icon="📈",
    layout="wide"
)

st.title("📈 분류 모델 성능 높이기")
st.write(
    "뇌졸중을 겪은 사람이 적은 데이터에서 "
    "클래스 가중치를 적용하기 전과 적용한 후의 결과를 비교합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "tkp12301/data/refs/heads/main/stroke.csv"
)


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")

    data = data.sort_values("id").reset_index(drop=True)

    return data


try:
    data = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 3. 입력 속성 선택
# --------------------------------------------------

feature_names = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부"
}

default_features = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease"
]

st.subheader("1. 입력으로 사용할 속성 선택")

selected_features = st.multiselect(
    "분류 모델에 사용할 속성을 선택하세요.",
    options=list(feature_names.keys()),
    default=default_features,
    format_func=lambda x: feature_names[x]
)

if len(selected_features) < 2:
    st.warning("비교와 시각화를 위해 속성을 2개 이상 선택해 주세요.")
    st.stop()

st.write(
    "선택한 속성: "
    + ", ".join(feature_names[x] for x in selected_features)
)


# --------------------------------------------------
# 4. 데이터 분할
# --------------------------------------------------

# ID 순서로 정렬한 뒤 10명 중 앞의 3명을 테스트용으로 사용
test_mask = data.index % 10 < 3

train_df = data.loc[~test_mask].copy().reset_index(drop=True)
test_df = data.loc[test_mask].copy().reset_index(drop=True)

# 정답 데이터
y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

# 훈련 데이터의 중앙값으로 결측값 대체
train_medians = train_df[selected_features].median()

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)

# 숫자형 데이터로 변환
X_train = X_train.astype(float)
X_test = X_test.astype(float)

st.subheader("2. 데이터 분할 결과")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("전체 데이터", f"{len(data):,}명")

with col2:
    st.metric("훈련용 데이터", f"{len(train_df):,}명")

with col3:
    st.metric("테스트용 데이터", f"{len(test_df):,}명")

st.write(
    f"훈련용 데이터의 실제 뇌졸중 양성은 "
    f"{int(y_train.sum()):,}명이며, "
    f"테스트용 데이터의 실제 뇌졸중 양성은 "
    f"{int(y_test.sum()):,}명입니다."
)

st.caption(
    "BMI를 포함해 선택한 속성에 결측값이 있으면 "
    "훈련용 데이터에서 계산한 해당 속성의 중앙값으로 대체합니다."
)


# --------------------------------------------------
# 5. 로지스틱 회귀 표준화
# --------------------------------------------------

# 훈련 데이터의 평균과 표준편차 사용
train_mean = X_train.mean()
train_std = X_train.std(ddof=0)

# 표준편차가 0인 속성은 1로 처리
train_std = train_std.replace(0, 1)

X_train_scaled = (X_train - train_mean) / train_std
X_test_scaled = (X_test - train_mean) / train_std


# --------------------------------------------------
# 6. 가중치 적용 전후 모델 학습
# --------------------------------------------------

@st.cache_resource
def train_models(
    X_train_scaled,
    X_test_scaled,
    X_train,
    X_test,
    y_train
):
    models = {}

    # 로지스틱 회귀: 가중치 적용 전
    logistic_before = LogisticRegression(
        max_iter=2000,
        random_state=42
    )
    logistic_before.fit(X_train_scaled, y_train)

    # 로지스틱 회귀: 가중치 적용 후
    logistic_after = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )
    logistic_after.fit(X_train_scaled, y_train)

    # 의사결정트리: 가중치 적용 전
    tree_before = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        random_state=42
    )
    tree_before.fit(X_train, y_train)

    # 의사결정트리: 가중치 적용 후
    tree_after = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    )
    tree_after.fit(X_train, y_train)

    models["로지스틱 회귀"] = {
        "before": logistic_before,
        "after": logistic_after,
        "scaled": True
    }

    models["의사결정트리"] = {
        "before": tree_before,
        "after": tree_after,
        "scaled": False
    }

    return models


models = train_models(
    X_train_scaled,
    X_test_scaled,
    X_train,
    X_test,
    y_train
)


# --------------------------------------------------
# 7. 예측 함수
# --------------------------------------------------

def get_prediction(model, X, scaled=False):
    if scaled:
        X_input = (X - train_mean) / train_std
    else:
        X_input = X

    return model.predict(X_input)


def get_probability(model, X, scaled=False):
    if scaled:
        X_input = (X - train_mean) / train_std
    else:
        X_input = X

    return model.predict_proba(X_input)[:, 1]


# --------------------------------------------------
# 8. 혼동행렬 및 평가 지표 계산
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else None

    actual_positive = tp + fn
    predicted_positive = tp + fp

    recall = (
        tp / actual_positive
        if actual_positive > 0
        else None
    )

    precision = (
        tp / predicted_positive
        if predicted_positive > 0
        else None
    )

    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1 점수": f1
    }


def format_metric(value):
    if value is None:
        return "계산할 수 없음"

    return f"{value * 100:.2f}%"


# --------------------------------------------------
# 9. 테스트 데이터 예측
# --------------------------------------------------

results = {}

for model_name, model_info in models.items():

    before_pred = get_prediction(
        model_info["before"],
        X_test_scaled if model_info["scaled"] else X_test,
        scaled=False
    )

    after_pred = get_prediction(
        model_info["after"],
        X_test_scaled if model_info["scaled"] else X_test,
        scaled=False
    )

    before_metrics = calculate_metrics(y_test, before_pred)
    after_metrics = calculate_metrics(y_test, after_pred)

    results[model_name] = {
        "before_pred": before_pred,
        "after_pred": after_pred,
        "before": before_metrics,
        "after": after_metrics
    }


# --------------------------------------------------
# 10. 평가 결과 표
# --------------------------------------------------

st.divider()
st.subheader("3. 가중치 적용 전후 성능 비교")

st.write(
    "가중치 적용 전에는 두 종류의 데이터가 동일한 중요도로 학습됩니다. "
    "가중치 적용 후에는 적은 수의 뇌졸중 양성 데이터를 "
    "상대적으로 더 중요하게 학습하도록 조정합니다."
)

display_names = {
    "로지스틱 회귀": "로지스틱 회귀(확률로 답하는 모델)",
    "의사결정트리": "의사결정트리(질문으로 답하는 모델)"
}


def make_comparison_table(before, after):

    rows = [
        ("실제 음성 · 예측 음성 (TN)", before["TN"], after["TN"]),
        ("실제 음성 · 예측 양성 (FP)", before["FP"], after["FP"]),
        ("실제 양성 · 예측 음성 (FN)", before["FN"], after["FN"]),
        ("실제 양성 · 예측 양성 (TP)", before["TP"], after["TP"]),
        ("정확도", format_metric(before["정확도"]), format_metric(after["정확도"])),
        ("재현율", format_metric(before["재현율"]), format_metric(after["재현율"])),
        ("정밀도", format_metric(before["정밀도"]), format_metric(after["정밀도"])),
        ("F1 점수", format_metric(before["F1 점수"]), format_metric(after["F1 점수"]))
    ]

    return pd.DataFrame(
        rows,
        columns=["평가 항목", "가중치 적용 전", "가중치 적용 후"]
    )


def change_text(before, after):
    return (
        f"찾아낸 환자 수: {before['TP']}명 → {after['TP']}명"
        f" | 놓친 환자 수: {before['FN']}명 → {after['FN']}명"
        f" | 헛짚은 사람 수: {before['FP']}명 → {after['FP']}명"
    )


col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**{display_names['로지스틱 회귀']}**")

    st.dataframe(
        make_comparison_table(
            results["로지스틱 회귀"]["before"],
            results["로지스틱 회귀"]["after"]
        ),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        change_text(
            results["로지스틱 회귀"]["before"],
            results["로지스틱 회귀"]["after"]
        )
    )

with col2:
    st.markdown(f"**{display_names['의사결정트리']}**")

    st.dataframe(
        make_comparison_table(
            results["의사결정트리"]["before"],
            results["의사결정트리"]["after"]
        ),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        change_text(
            results["의사결정트리"]["before"],
            results["의사결정트리"]["after"]
        )
    )


st.caption(
    "TN: 실제 음성을 음성으로 예측한 사람, "
    "FP: 실제 음성을 양성으로 예측한 사람, "
    "FN: 실제 양성을 음성으로 예측한 사람, "
    "TP: 실제 양성을 양성으로 예측한 사람"
)


# --------------------------------------------------
# 11. 2차원 또는 3차원 산점도 축 선택
# --------------------------------------------------

st.divider()
st.subheader("4. 가중치 적용 전후의 결정 경계 비교")

st.write(
    "점의 색은 실제 뇌졸중 여부를 나타내며, "
    "경계선 또는 경계면은 모델이 양성으로 판단할 확률이 "
    "0.5가 되는 위치입니다."
)

if len(selected_features) == 2:

    x_feature = selected_features[0]
    y_feature = selected_features[1]

    st.write(
        f"가로축: {feature_names[x_feature]} / "
        f"세로축: {feature_names[y_feature]}"
    )

    plot_data = test_df.copy()
    plot_data["실제 뇌졸중 여부"] = y_test.map({
        0: "뇌졸중 없음",
        1: "뇌졸중 있음"
    })

    fig = px.scatter(
        plot_data,
        x=x_feature,
        y=y_feature,
        color="실제 뇌졸중 여부",
        color_discrete_map={
            "뇌졸중 없음": "#2878B5",
            "뇌졸중 있음": "#E45756"
        },
        labels={
            x_feature: feature_names[x_feature],
            y_feature: feature_names[y_feature]
        },
        title="실제 뇌졸중 여부와 가중치 적용 전후 결정 경계"
    )

    # 나머지 선택 속성은 테스트 데이터 중앙값으로 고정
    base_values = X_train.median()

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

    x_grid_values = [
        x_min + (x_max - x_min) * i / 59
        for i in range(60)
    ]

    y_grid_values = [
        y_min + (y_max - y_min) * i / 59
        for i in range(60)
    ]

    grid_rows = []

    for y_value in y_grid_values:
        for x_value in x_grid_values:
            row = base_values.copy()
            row[x_feature] = x_value
            row[y_feature] = y_value
            grid_rows.append(row)

    grid_df = pd.DataFrame(grid_rows, columns=selected_features)

    boundary_found = {}

    for model_name in ["로지스틱 회귀", "의사결정트리"]:

        model_info = models[model_name]

        for stage, color, label in [
            ("before", "#F39C12", "가중치 적용 전"),
            ("after", "#8E44AD", "가중치 적용 후")
        ]:

            probability = get_probability(
                model_info[stage],
                grid_df,
                scaled=model_info["scaled"]
            )

            probability_grid = [
                probability[i * len(x_grid_values):
                             (i + 1) * len(x_grid_values)]
                for i in range(len(y_grid_values))
            ]

            min_probability = min(probability)
            max_probability = max(probability)

            found = min_probability <= 0.5 <= max_probability

            boundary_found[(model_name, stage)] = found

            if found:
                fig.add_trace(
                    go.Contour(
                        x=x_grid_values,
                        y=y_grid_values,
                        z=probability_grid,
                        contours=dict(
                            start=0.5,
                            end=0.5,
                            size=0.5,
                            coloring="none"
                        ),
                        line=dict(
                            color=color,
                            width=3,
                            dash="solid" if stage == "before" else "dash"
                        ),
                        showscale=False,
                        hoverinfo="skip",
                        name=f"{display_names[model_name]} - {label}"
                    )
                )

    fig.update_layout(
        xaxis_title=feature_names[x_feature],
        yaxis_title=feature_names[y_feature],
        legend_title="실제 여부 및 결정 경계",
        height=650
    )

    st.plotly_chart(fig, use_container_width=True)

    for model_name in ["로지스틱 회귀", "의사결정트리"]:
        for stage, label in [
            ("before", "가중치 적용 전"),
            ("after", "가중치 적용 후")
        ]:
            if not boundary_found[(model_name, stage)]:
                st.info(
                    f"{display_names[model_name]}의 {label} "
                    "0.5 경계선은 현재 그래프 범위 안에 나타나지 않습니다."
                )


else:

    axis_options = selected_features

    st.write("3차원 산점도에 사용할 세 가지 축을 선택하세요.")

    axis_col1, axis_col2, axis_col3 = st.columns(3)

    with axis_col1:
        x_feature = st.selectbox(
            "가로축(X)",
            axis_options,
            index=0,
            format_func=lambda x: feature_names[x]
        )

    with axis_col2:
        y_default = 1 if len(axis_options) > 1 else 0
        y_feature = st.selectbox(
            "세로축(Y)",
            axis_options,
            index=y_default,
            format_func=lambda x: feature_names[x]
        )

    with axis_col3:
        z_default = 2 if len(axis_options) > 2 else 0
        z_feature = st.selectbox(
            "높이축(Z)",
            axis_options,
            index=z_default,
            format_func=lambda x: feature_names[x]
        )

    if len({x_feature, y_feature, z_feature}) != 3:
        st.warning("3차원 그래프의 세 축은 서로 다른 속성을 선택해야 합니다.")
        st.stop()

    plot_data = test_df.copy()
    plot_data["실제 뇌졸중 여부"] = y_test.map({
        0: "뇌졸중 없음",
        1: "뇌졸중 있음"
    })

    fig = px.scatter_3d(
        plot_data,
        x=x_feature,
        y=y_feature,
        z=z_feature,
        color="실제 뇌졸중 여부",
        color_discrete_map={
            "뇌졸중 없음": "#2878B5",
            "뇌졸중 있음": "#E45756"
        },
        labels={
            x_feature: feature_names[x_feature],
            y_feature: feature_names[y_feature],
            z_feature: feature_names[z_feature]
        },
        title="실제 뇌졸중 여부와 가중치 적용 전후 0.5 경계면"
    )

    # 선택한 세 속성 이외의 속성은 훈련 데이터 중앙값으로 고정
    base_values = X_train.median()

    x_min = float(X_test[x_feature].min())
    x_max = float(X_test[x_feature].max())
    y_min = float(X_test[y_feature].min())
    y_max = float(X_test[y_feature].max())
    z_min = float(X_test[z_feature].min())
    z_max = float(X_test[z_feature].max())

    if x_min == x_max:
        x_min -= 1
        x_max += 1

    if y_min == y_max:
        y_min -= 1
        y_max += 1

    if z_min == z_max:
        z_min -= 1
        z_max += 1

    grid_size = 18

    x_values = [
        x_min + (x_max - x_min) * i / (grid_size - 1)
        for i in range(grid_size)
    ]

    y_values = [
        y_min + (y_max - y_min) * i / (grid_size - 1)
        for i in range(grid_size)
    ]

    z_values = [
        z_min + (z_max - z_min) * i / (grid_size - 1)
        for i in range(grid_size)
    ]

    grid_rows = []

    for z_value in z_values:
        for y_value in y_values:
            for x_value in x_values:
                row = base_values.copy()
                row[x_feature] = x_value
                row[y_feature] = y_value
                row[z_feature] = z_value
                grid_rows.append(row)

    grid_df = pd.DataFrame(grid_rows, columns=selected_features)

    for model_name in ["로지스틱 회귀", "의사결정트리"]:

        model_info = models[model_name]

        for stage, color, label in [
            ("before", "#F39C12", "가중치 적용 전"),
            ("after", "#8E44AD", "가중치 적용 후")
        ]:

            probability = get_probability(
                model_info[stage],
                grid_df,
                scaled=model_info["scaled"]
            )

            min_probability = float(min(probability))
            max_probability = float(max(probability))

            if not (min_probability <= 0.5 <= max_probability):
                st.info(
                    f"{display_names[model_name]}의 {label} "
                    "0.5 경계면은 현재 그래프 범위 안에 나타나지 않습니다."
                )
                continue

            fig.add_trace(
                go.Isosurface(
                    x=[
                        row[x_feature]
                        for row in grid_rows
                    ],
                    y=[
                        row[y_feature]
                        for row in grid_rows
                    ],
                    z=[
                        row[z_feature]
                        for row in grid_rows
                    ],
                    value=probability,
                    isomin=0.5,
                    isomax=0.5,
                    surface_count=1,
                    colorscale=[
                        [0, color],
                        [1, color]
                    ],
                    opacity=0.35,
                    caps=dict(
                        x_show=False,
                        y_show=False,
                        z_show=False
                    ),
                    showscale=False,
                    name=f"{display_names[model_name]} - {label}",
                    hoverinfo="skip"
                )
            )

    fig.update_layout(
        scene=dict(
            xaxis_title=feature_names[x_feature],
            yaxis_title=feature_names[y_feature],
            zaxis_title=feature_names[z_feature]
        ),
        height=750,
        legend_title="실제 여부 및 결정 경계"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "마우스로 그래프를 드래그하면 회전할 수 있으며, "
        "선택한 세 축 이외의 입력 속성은 훈련 데이터 중앙값으로 고정해 "
        "경계면을 계산합니다."
    )


# --------------------------------------------------
# 12. 결과 해석 안내
# --------------------------------------------------

st.divider()
st.subheader("5. 결과 해석하기")

st.markdown(
    """
    - **재현율**은 실제 뇌졸중 환자 중 모델이 찾아낸 환자의 비율입니다.
    - **정밀도**는 모델이 뇌졸중 환자라고 예측한 사람 중 실제 환자의 비율입니다.
    - **정확도**는 전체 테스트 데이터 중 예측이 맞은 비율입니다.
    - **F1 점수**는 정밀도와 재현율의 조화평균으로, 두 지표를 함께 고려합니다.

    클래스 가중치를 적용하면 적은 수의 양성 데이터를 더 중요하게 학습하므로
    재현율이 높아질 수 있습니다. 다만 양성으로 잘못 예측하는 사람도
    늘어날 수 있어 정밀도와 정확도가 낮아질 가능성도 있습니다.

    따라서 정확도 하나만으로 모델의 성능을 판단하지 않고,
    찾아낸 환자 수와 놓친 환자 수, 헛짚은 사람 수를 함께 비교해야 합니다.
    """
)

st.warning(
    "이 페이지는 머신러닝 분류 원리를 학습하기 위한 실습입니다. "
    "실제 뇌졸중 진단이나 개인의 질병 위험 판단에 사용할 수 있는 의료 도구가 아닙니다."
)
