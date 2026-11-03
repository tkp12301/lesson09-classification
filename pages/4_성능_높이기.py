
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------

st.set_page_config(
    page_title="성능 높이기",
    page_icon="📈",
    layout="wide"
)

st.title("📈 분류 모델 성능 높이기")

st.write(
    "클래스 가중치를 적용하기 전과 적용한 후의 "
    "분류 모델 성능 및 0.5 결정 경계를 비교합니다."
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
    st.error(f"데이터를 불러오지 못했습니다: {e}")
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

st.subheader("1. 입력 속성 선택")

selected_features = st.multiselect(
    "분류 모델에 사용할 속성을 선택하세요.",
    options=list(feature_names.keys()),
    default=default_features,
    format_func=lambda x: feature_names[x]
)

if len(selected_features) < 2:
    st.warning("속성을 2개 이상 선택해 주세요.")
    st.stop()

st.write(
    "선택한 속성: "
    + ", ".join(feature_names[x] for x in selected_features)
)


# --------------------------------------------------
# 4. 데이터 분할
# --------------------------------------------------

# ID 순 정렬 후 10명 중 앞의 3명을 테스트 데이터로 사용
test_mask = data.index % 10 < 3

train_df = data.loc[~test_mask].copy().reset_index(drop=True)
test_df = data.loc[test_mask].copy().reset_index(drop=True)

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

# 훈련 데이터 중앙값으로 결측값 대체
train_medians = train_df[selected_features].median()

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

X_train = X_train.fillna(train_medians).astype(float)
X_test = X_test.fillna(train_medians).astype(float)


st.subheader("2. 데이터 분할 결과")

col1, col2, col3 = st.columns(3)

col1.metric("전체 데이터", f"{len(data):,}명")
col2.metric("훈련용 데이터", f"{len(train_df):,}명")
col3.metric("테스트용 데이터", f"{len(test_df):,}명")

st.write(
    f"훈련용 실제 양성: {int(y_train.sum()):,}명 | "
    f"테스트용 실제 양성: {int(y_test.sum()):,}명"
)

st.caption(
    "선택한 속성의 결측값은 훈련용 데이터의 중앙값으로 대체합니다."
)


# --------------------------------------------------
# 5. 로지스틱 회귀 표준화
# --------------------------------------------------

train_mean = X_train.mean()
train_std = X_train.std(ddof=0)

train_std = train_std.replace(0, 1)

X_train_scaled = (X_train - train_mean) / train_std
X_test_scaled = (X_test - train_mean) / train_std


# --------------------------------------------------
# 6. 모델 학습
# --------------------------------------------------

# 가중치 적용 전
logistic_before = LogisticRegression(
    max_iter=2000,
    random_state=42
)

tree_before = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

# 가중치 적용 후
logistic_after = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

tree_after = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

logistic_before.fit(X_train_scaled, y_train)
logistic_after.fit(X_train_scaled, y_train)

tree_before.fit(X_train, y_train)
tree_after.fit(X_train, y_train)


models = {
    "로지스틱 회귀": {
        "before": logistic_before,
        "after": logistic_after,
        "scaled": True
    },
    "의사결정트리": {
        "before": tree_before,
        "after": tree_after,
        "scaled": False
    }
}


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
# 8. 평가 지표 계산
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else None

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
        X_test,
        scaled=model_info["scaled"]
    )

    after_pred = get_prediction(
        model_info["after"],
        X_test,
        scaled=model_info["scaled"]
    )

    results[model_name] = {
        "before_pred": before_pred,
        "after_pred": after_pred,
        "before": calculate_metrics(y_test, before_pred),
        "after": calculate_metrics(y_test, after_pred)
    }


# --------------------------------------------------
# 10. 성능 비교표
# --------------------------------------------------

st.divider()
st.subheader("3. 가중치 적용 전후 성능 비교")

display_names = {
    "로지스틱 회귀": "로지스틱 회귀(확률로 답하는 모델)",
    "의사결정트리": "의사결정트리(질문으로 답하는 모델)"
}


def make_comparison_table(before, after):

    rows = [
        ("TN 실제 음성·예측 음성", before["TN"], after["TN"]),
        ("FP 실제 음성·예측 양성", before["FP"], after["FP"]),
        ("FN 실제 양성·예측 음성", before["FN"], after["FN"]),
        ("TP 실제 양성·예측 양성", before["TP"], after["TP"]),
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
    "TN: 실제 음성을 음성으로 예측 | "
    "FP: 실제 음성을 양성으로 예측 | "
    "FN: 실제 양성을 음성으로 예측 | "
    "TP: 실제 양성을 양성으로 예측"
)


# --------------------------------------------------
# 11. 결정 경계 계산에 사용할 공통 함수
# --------------------------------------------------

def make_grid(x_values, y_values, z_values=None):

    base_values = X_train.median()
    rows = []

    if z_values is None:

        for y_value in y_values:
            for x_value in x_values:

                row = base_values.copy()
                row[x_feature] = x_value
                row[y_feature] = y_value

                rows.append(row)

    else:

        for z_value in z_values:
            for y_value in y_values:
                for x_value in x_values:

                    row = base_values.copy()
                    row[x_feature] = x_value
                    row[y_feature] = y_value
                    row[z_feature] = z_value

                    rows.append(row)

    return pd.DataFrame(rows, columns=selected_features)


def make_range(series):

    minimum = float(series.min())
    maximum = float(series.max())

    if minimum == maximum:
        minimum -= 1
        maximum += 1

    return minimum, maximum


# --------------------------------------------------
# 12. 결정 경계 시각화
# --------------------------------------------------

st.divider()
st.subheader("4. 실제 뇌졸중 여부와 0.5 결정 경계")

st.write(
    "점의 색은 실제 뇌졸중 여부를 나타냅니다. "
    "경계선 또는 경계면은 모델이 뇌졸중일 확률을 0.5로 예측하는 위치입니다."
)

# 그래프 색상
boundary_styles = {
    ("로지스틱 회귀", "before"): {
        "color": "#F39C12",
        "dash": "solid",
        "label": "로지스틱 회귀 - 가중치 적용 전"
    },
    ("로지스틱 회귀", "after"): {
        "color": "#8E44AD",
        "dash": "dash",
        "label": "로지스틱 회귀 - 가중치 적용 후"
    },
    ("의사결정트리", "before"): {
        "color": "#16A085",
        "dash": "solid",
        "label": "의사결정트리 - 가중치 적용 전"
    },
    ("의사결정트리", "after"): {
        "color": "#E74C3C",
        "dash": "dash",
        "label": "의사결정트리 - 가중치 적용 후"
    }
}


# --------------------------------------------------
# 13. 2차원 산점도와 경계선
# --------------------------------------------------

if len(selected_features) == 2:

    x_feature = selected_features[0]
    y_feature = selected_features[1]

    st.markdown("### 2차원 산점도")

    st.write(
        f"가로축: {feature_names[x_feature]} | "
        f"세로축: {feature_names[y_feature]}"
    )

    # 결측값 대체가 완료된 X_test 사용
    plot_data = X_test.copy()

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
        title="실제 뇌졸중 여부와 가중치 적용 전후 0.5 경계선"
    )

    x_min, x_max = make_range(X_test[x_feature])
    y_min, y_max = make_range(X_test[y_feature])

    x_values = [
        x_min + (x_max - x_min) * i / 119
        for i in range(120)
    ]

    y_values = [
        y_min + (y_max - y_min) * i / 119
        for i in range(120)
    ]

    grid_df = make_grid(x_values, y_values)

    boundary_status = {}

    # 네 가지 경계선을 모두 계산
    for model_name in ["로지스틱 회귀", "의사결정트리"]:

        model_info = models[model_name]

        for stage in ["before", "after"]:

            model = model_info[stage]

            probability = get_probability(
                model,
                grid_df,
                scaled=model_info["scaled"]
            )

            # y축 행, x축 열로 확률 배열 구성
            probability_grid = [
                probability[
                    i * len(x_values):(i + 1) * len(x_values)
                ]
                for i in range(len(y_values))
            ]

            minimum_probability = float(min(probability))
            maximum_probability = float(max(probability))

            # 그래프 영역 안에 0.5 경계가 존재하는지 확인
            has_boundary = (
                minimum_probability <= 0.5
                and maximum_probability >= 0.5
            )

            boundary_status[(model_name, stage)] = has_boundary

            if has_boundary:

                style = boundary_styles[(model_name, stage)]

                # 0.5 등고선을 직접 표시
                fig.add_trace(
                    go.Contour(
                        x=x_values,
                        y=y_values,
                        z=probability_grid,
                        contours=dict(
                            type="constraint",
                            operation="=",
                            value=0.5
                        ),
                        line=dict(
                            color=style["color"],
                            width=4,
                            dash=style["dash"]
                        ),
                        showscale=False,
                        hovertemplate=(
                            f"{style['label']}<br>"
                            "예측 확률: 0.5"
                            "<extra></extra>"
                        ),
                        name=style["label"],
                        showlegend=True
                    )
                )

    fig.update_layout(
        xaxis_title=feature_names[x_feature],
        yaxis_title=feature_names[y_feature],
        height=700,
        legend_title="실제 여부 및 결정 경계",
        hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 0.5 경계선 표시 결과")

    for model_name in ["로지스틱 회귀", "의사결정트리"]:

        for stage, stage_name in [
            ("before", "가중치 적용 전"),
            ("after", "가중치 적용 후")
        ]:

            if boundary_status[(model_name, stage)]:
                st.success(
                    f"{display_names[model_name]}의 {stage_name}: "
                    "0.5 경계선이 그래프에 표시되었습니다."
                )
            else:
                st.warning(
                    f"{display_names[model_name]}의 {stage_name}: "
                    "0.5 경계선이 현재 그래프 범위 안에 없습니다."
                )


# --------------------------------------------------
# 14. 3차원 산점도와 경계면
# --------------------------------------------------

else:

    st.markdown("### 3차원 산점도")

    st.write("3차원 그래프의 가로축, 세로축, 높이축을 선택하세요.")

    axis_col1, axis_col2, axis_col3 = st.columns(3)

    with axis_col1:

        x_feature = st.selectbox(
            "가로축(X)",
            selected_features,
            index=0,
            format_func=lambda x: feature_names[x]
        )

    with axis_col2:

        y_feature = st.selectbox(
            "세로축(Y)",
            selected_features,
            index=1,
            format_func=lambda x: feature_names[x]
        )

    with axis_col3:

        z_feature = st.selectbox(
            "높이축(Z)",
            selected_features,
            index=2,
            format_func=lambda x: feature_names[x]
        )

    if len({x_feature, y_feature, z_feature}) != 3:

        st.warning("세 축은 서로 다른 속성을 선택해야 합니다.")
        st.stop()

    plot_data = X_test.copy()

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

    x_min, x_max = make_range(X_test[x_feature])
    y_min, y_max = make_range(X_test[y_feature])
    z_min, z_max = make_range(X_test[z_feature])

    grid_size = 25

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

    grid_df = make_grid(x_values, y_values, z_values)

    boundary_status = {}

    for model_name in ["로지스틱 회귀", "의사결정트리"]:

        model_info = models[model_name]

        for stage in ["before", "after"]:

            model = model_info[stage]

            probability = get_probability(
                model,
                grid_df,
                scaled=model_info["scaled"]
            )

            # 확률값을 z축 방향으로 묶음
            probability_volume = []

            index = 0

            for z_index in range(len(z_values)):

                z_layer = []

                for y_index in range(len(y_values)):

                    row = probability[
                        index:index + len(x_values)
                    ]

                    z_layer.append(row)
                    index += len(x_values)

                probability_volume.append(z_layer)

            # 0.5 경계면 계산
            # 각 x-y 위치에서 z 방향으로 확률이 0.5가 되는 지점을 찾음
            surface_x = []
            surface_y = []
            surface_z = []

            for y_index, y_value in enumerate(y_values):

                surface_row = []

                for x_index, x_value in enumerate(x_values):

                    crossing_height = None

                    for z_index in range(len(z_values) - 1):

                        p1 = probability_volume[z_index][y_index][x_index]
                        p2 = probability_volume[z_index + 1][y_index][x_index]

                        if p1 == 0.5:
                            crossing_height = z_values[z_index]
                            break

                        if (p1 - 0.5) * (p2 - 0.5) < 0:

                            ratio = (0.5 - p1) / (p2 - p1)

                            crossing_height = (
                                z_values[z_index]
                                + ratio * (
                                    z_values[z_index + 1]
                                    - z_values[z_index]
                                )
                            )

                            break

                    if crossing_height is not None:
                        surface_row.append(crossing_height)
                    else:
                        surface_row.append(float("nan"))

                surface_z.append(surface_row)

            # 경계면이 하나라도 계산되었는지 확인
            has_boundary = any(
                not pd.isna(value)
                for row in surface_z
                for value in row
            )

            boundary_status[(model_name, stage)] = has_boundary

            if has_boundary:

                style = boundary_styles[(model_name, stage)]

                fig.add_trace(
                    go.Surface(
                        x=x_values,
                        y=y_values,
                        z=surface_z,
                        colorscale=[
                            [0, style["color"]],
                            [1, style["color"]]
                        ],
                        opacity=0.55,
                        showscale=False,
                        connectgaps=False,
                        name=style["label"],
                        showlegend=True,
                        hovertemplate=(
                            f"{style['label']}<br>"
                            "예측 확률: 0.5"
                            "<extra></extra>"
                        )
                    )
                )

    fig.update_layout(
        scene=dict(
            xaxis_title=feature_names[x_feature],
            yaxis_title=feature_names[y_feature],
            zaxis_title=feature_names[z_feature]
        ),
        height=800,
        legend_title="실제 여부 및 결정 경계"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "그래프를 마우스로 드래그하면 회전할 수 있습니다. "
        "경계면은 선택한 세 속성으로 구성한 공간에서 계산하며, "
        "나머지 속성은 훈련 데이터 중앙값으로 고정합니다."
    )

    st.markdown("#### 0.5 경계면 표시 결과")

    for model_name in ["로지스틱 회귀", "의사결정트리"]:

        for stage, stage_name in [
            ("before", "가중치 적용 전"),
            ("after", "가중치 적용 후")
        ]:

            if boundary_status[(model_name, stage)]:
                st.success(
                    f"{display_names[model_name]}의 {stage_name}: "
                    "0.5 경계면이 그래프에 표시되었습니다."
                )
            else:
                st.warning(
                    f"{display_names[model_name]}의 {stage_name}: "
                    "현재 그래프 범위에서 0.5 경계면을 찾지 못했습니다."
                )


# --------------------------------------------------
# 15. 결과 해석
# --------------------------------------------------

st.divider()
st.subheader("5. 결과 해석")

st.markdown(
    """
    - **정확도**: 전체 테스트 데이터 중 올바르게 예측한 비율
    - **재현율**: 실제 뇌졸중 환자 중 모델이 찾아낸 비율
    - **정밀도**: 모델이 뇌졸중 환자라고 예측한 사람 중 실제 환자의 비율
    - **F1 점수**: 정밀도와 재현율의 조화평균

    클래스 가중치를 적용하면 적은 수의 양성 데이터를 더 중요하게
    학습하도록 조정됩니다. 따라서 실제 환자를 찾아내는 재현율이
    높아질 수 있지만, 음성을 양성으로 잘못 예측하는 경우도
    증가할 수 있습니다.

    0.5 경계선은 모델의 예측 확률이 0.5인 위치입니다.
    경계선의 이동은 가중치 적용에 따라 모델의 분류 기준이
    어떻게 달라졌는지를 보여 줍니다.
    """
)

st.warning(
    "이 페이지는 머신러닝 분류 원리를 학습하기 위한 실습입니다. "
    "실제 뇌졸중 진단이나 개인의 질병 위험 판단에 사용할 수 없습니다."
)
