
import streamlit as st
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ==================================================
# 1. 기본 설정
# ==================================================

st.set_page_config(
    page_title="뇌졸중 예측 채점표",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 뇌졸중 예측 채점표")
st.write(
    "테스트 데이터의 실제 결과와 모델의 예측 결과를 비교하여 "
    "혼동행렬과 분류 성능 지표를 확인합니다."
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

MODEL_LOGISTIC = "로지스틱 회귀(확률로 답하는 모델)"
MODEL_TREE = "의사결정트리(질문으로 답하는 모델)"
MODEL_BASELINE = "훈련용에서 많은 쪽으로만 답하는 모델"


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
    "분류 모델 학습에 사용할 속성을 선택하세요.",
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
    "**선택한 속성:** "
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

# 목록에 표시할 속성도 숫자형으로 변환
display_columns = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease"
]

for col in display_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ID 번호 순서로 정렬
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
    "ID 순서로 정렬한 뒤 10명씩 묶어 앞의 3명은 테스트용, "
    "나머지 7명은 훈련용으로 분리합니다."
)

test_mask = [
    i % 10 < 3
    for i in range(len(df))
]

train_mask = [
    not value
    for value in test_mask
]

train_df = df.loc[train_mask].copy().reset_index(drop=True)
test_df = df.loc[test_mask].copy().reset_index(drop=True)

st.write(f"- 전체 분석 데이터: **{len(df):,}명**")
st.write(f"- 훈련용 데이터: **{len(train_df):,}명**")
st.write(f"- 테스트용 데이터: **{len(test_df):,}명**")

if len(train_df) == 0 or len(test_df) == 0:
    st.error("훈련용 또는 테스트용 데이터가 없습니다.")
    st.stop()

if train_df["stroke"].nunique() != 2:
    st.error(
        "훈련용 데이터에 뇌졸중 여부 두 종류가 모두 있어야 합니다."
    )
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
    "선택한 속성에 결측값이 있으면 훈련용 데이터의 중앙값을 계산해 "
    "훈련용과 테스트용 데이터에 동일하게 적용합니다."
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
                f"{FEATURE_NAMES[col]} 속성의 훈련용 데이터가 "
                "모두 결측값입니다."
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
    st.info("선택한 속성에 처리할 결측값이 없습니다.")

if X_train.isna().any().any() or X_test.isna().any().any():
    st.error("결측값 처리가 완료되지 않았습니다.")
    st.stop()

st.write(
    f"- 훈련용 데이터의 남은 결측값: "
    f"**{int(X_train.isna().sum().sum())}개**"
)
st.write(
    f"- 테스트용 데이터의 남은 결측값: "
    f"**{int(X_test.isna().sum().sum())}개**"
)


# ==================================================
# 7. 로지스틱 회귀 표준화
# ==================================================

st.subheader("④ 데이터 크기 맞추기(표준화)")

st.write(
    "로지스틱 회귀는 훈련용 데이터의 평균과 표준편차를 이용해 "
    "속성별 데이터 크기를 맞춥니다. 테스트용 데이터에도 "
    "훈련용 데이터에서 계산한 기준을 동일하게 적용합니다."
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


# ==================================================
# 8. 분류 모델 학습
# ==================================================

st.header("3. 분류 모델 학습")

st.write(
    "훈련용 데이터만 사용하여 로지스틱 회귀와 의사결정트리를 "
    "학습합니다. 테스트용 데이터는 모델 학습에 사용하지 않습니다."
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

st.write("**학습된 모델**")

training_table = pd.DataFrame({
    "모델": [
        MODEL_LOGISTIC,
        MODEL_TREE
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

st.write("- 로지스틱 회귀: 최대 반복 횟수 2,000회")
st.write("- 의사결정트리: 최대 질문 횟수 3회, 마지막 마디 최소 인원 5명")
st.write("- 두 모델의 난수 고정값: 42")


# ==================================================
# 9. 테스트 데이터 예측
# ==================================================

st.header("4. 테스트 데이터 예측")

# 로지스틱 회귀 예측
logistic_predictions = logistic_model.predict(
    X_test_scaled
)

# 의사결정트리 예측
tree_predictions = tree_model.predict(
    X_test
)

# 기준 모델: 훈련용 데이터에서 많은 쪽만 예측
majority_class = int(y_train.value_counts().idxmax())

baseline_predictions = [
    majority_class for _ in range(len(y_test))
]

# 테스트 데이터의 실제 결과
actual_values = y_test.tolist()

# 예측 결과를 정수 리스트로 통일
prediction_sets = {
    MODEL_LOGISTIC: [int(x) for x in logistic_predictions],
    MODEL_TREE: [int(x) for x in tree_predictions],
    MODEL_BASELINE: [int(x) for x in baseline_predictions]
}

st.write(
    f"**테스트용 사람 수: {len(test_df):,}명 / "
    f"그중 실제 뇌졸중인 사람 수: {sum(actual_values):,}명**"
)


# ==================================================
# 10. 혼동행렬 계산 함수
# ==================================================

def calculate_confusion(actual, predicted):
    """
    실제 결과와 예측 결과로 TP, FP, FN, TN 계산

    TP: 실제 1, 예측 1
    FP: 실제 0, 예측 1
    FN: 실제 1, 예측 0
    TN: 실제 0, 예측 0
    """

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for real, pred in zip(actual, predicted):
        if real == 1 and pred == 1:
            tp += 1
        elif real == 0 and pred == 1:
            fp += 1
        elif real == 1 and pred == 0:
            fn += 1
        elif real == 0 and pred == 0:
            tn += 1

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    }


# ==================================================
# 11. 성능 지표 계산 함수
# ==================================================

def calculate_metrics(confusion):
    tp = confusion["TP"]
    fp = confusion["FP"]
    fn = confusion["FN"]
    tn = confusion["TN"]

    total = tp + fp + fn + tn

    # 정확도 = (TP + TN) / 전체 인원
    accuracy = (
        (tp + tn) / total
        if total > 0 else None
    )

    # 재현율 = TP / 실제 뇌졸중인 사람 수
    recall_denominator = tp + fn

    recall = (
        tp / recall_denominator
        if recall_denominator > 0 else None
    )

    # 정밀도 = TP / 뇌졸중이라고 예측한 사람 수
    precision_denominator = tp + fp

    precision = (
        tp / precision_denominator
        if precision_denominator > 0 else None
    )

    # F1 = 2 * 정밀도 * 재현율 / (정밀도 + 재현율)
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0
    else:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    return {
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1
    }


def format_metric(value):
    if value is None:
        return "계산할 수 없음"

    return f"{value * 100:.2f}%"


# ==================================================
# 12. 각 모델의 혼동행렬과 성능 지표
# ==================================================

st.header("5. 모델별 혼동행렬")

confusion_results = {}
metric_results = {}

for model_name, predictions in prediction_sets.items():
    confusion = calculate_confusion(
        actual_values,
        predictions
    )

    metrics = calculate_metrics(confusion)

    confusion_results[model_name] = confusion
    metric_results[model_name] = metrics


# 혼동행렬 표 만들기
def make_confusion_table(confusion):
    tp = confusion["TP"]
    fp = confusion["FP"]
    fn = confusion["FN"]
    tn = confusion["TN"]

    actual_negative_total = tn + fp
    actual_positive_total = fn + tp

    predicted_negative_total = tn + fn
    predicted_positive_total = fp + tp

    total = tn + fp + fn + tp

    table = pd.DataFrame(
        [
            [
                f"TN: {tn:,}명",
                f"FP: {fp:,}명",
                f"{actual_negative_total:,}명"
            ],
            [
                f"FN: {fn:,}명",
                f"TP: {tp:,}명",
                f"{actual_positive_total:,}명"
            ],
            [
                f"{predicted_negative_total:,}명",
                f"{predicted_positive_total:,}명",
                f"{total:,}명"
            ]
        ],
        index=[
            "실제 0 (뇌졸중 아님)",
            "실제 1 (뇌졸중)",
            "열 합계"
        ],
        columns=[
            "예측 0 (뇌졸중 아님)",
            "예측 1 (뇌졸중)",
            "행 합계"
        ]
    )

    return table


# 각 모델의 혼동행렬을 나란히 표시
matrix_col1, matrix_col2 = st.columns(2)

with matrix_col1:
    st.subheader(MODEL_LOGISTIC)
    st.dataframe(
        make_confusion_table(
            confusion_results[MODEL_LOGISTIC]
        ),
        use_container_width=True
    )

with matrix_col2:
    st.subheader(MODEL_TREE)
    st.dataframe(
        make_confusion_table(
            confusion_results[MODEL_TREE]
        ),
        use_container_width=True
    )

st.caption(
    "TN: 실제 뇌졸중 아님을 뇌졸중 아님으로 예측 / "
    "FP: 실제 뇌졸중 아님을 뇌졸중으로 예측 / "
    "FN: 실제 뇌졸중을 뇌졸중 아님으로 예측 / "
    "TP: 실제 뇌졸중을 뇌졸중으로 예측"
)


# ==================================================
# 13. 성능 지표 비교표
# ==================================================

st.header("6. 성능 지표 비교")

st.write(
    "혼동행렬의 네 가지 값으로 정확도, 재현율, 정밀도, F1을 계산합니다."
)

metric_table = pd.DataFrame([
    {
        "모델": model_name,
        "정확도": format_metric(metric_results[model_name]["정확도"]),
        "재현율": format_metric(metric_results[model_name]["재현율"]),
        "정밀도": format_metric(metric_results[model_name]["정밀도"]),
        "F1": format_metric(metric_results[model_name]["F1"])
    }
    for model_name in [
        MODEL_LOGISTIC,
        MODEL_TREE,
        MODEL_BASELINE
    ]
])

st.dataframe(
    metric_table,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "재현율 = TP / (TP + FN), "
    "정밀도 = TP / (TP + FP), "
    "F1 = 2 × 정밀도 × 재현율 / (정밀도 + 재현율)"
)

st.caption(
    "뇌졸중이라고 예측한 사람이 0명이면 정밀도와 F1은 "
    "'계산할 수 없음'으로 표시합니다."
)


# ==================================================
# 14. 놓친 환자와 헛짚은 사람 목록
# ==================================================

st.header("7. 잘못 예측한 사람 확인")

st.write(
    "모델을 선택하면 해당 모델이 잘못 예측한 테스트 데이터의 "
    "사람들을 확인할 수 있습니다."
)

selected_model = st.selectbox(
    "목록을 확인할 모델 선택",
    options=[
        MODEL_LOGISTIC,
        MODEL_TREE,
        MODEL_BASELINE
    ]
)

selected_predictions = prediction_sets[selected_model]
selected_confusion = confusion_results[selected_model]

# 테스트 데이터와 실제 결과, 예측 결과 결합
result_df = test_df.copy().reset_index(drop=True)

result_df["실제 결과"] = actual_values
result_df["예측 결과"] = selected_predictions

# FN: 실제 1인데 예측 0인 사람
missed_patients = result_df[
    (result_df["실제 결과"] == 1)
    & (result_df["예측 결과"] == 0)
].copy()

# FP: 실제 0인데 예측 1인 사람
false_alarm_patients = result_df[
    (result_df["실제 결과"] == 0)
    & (result_df["예측 결과"] == 1)
].copy()

# 나이가 많은 순서로 정렬
missed_patients = missed_patients.sort_values(
    by="age",
    ascending=False,
    na_position="last"
)

false_alarm_patients = false_alarm_patients.sort_values(
    by="age",
    ascending=False,
    na_position="last"
)


# 목록에 표시할 열과 한글 이름
list_columns = [
    "id",
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease"
]

list_column_names = {
    "id": "번호",
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부"
}


def prepare_person_list(data):
    display_data = data[list_columns].copy()

    display_data = display_data.rename(
        columns=list_column_names
    )

    display_data = display_data.rename(
        columns={"번호": "번호"}
    )

    display_data["번호"] = display_data["번호"].astype("Int64")

    display_data["나이"] = display_data["나이"].round(1)
    display_data["평균 혈당"] = (
        display_data["평균 혈당"].round(2)
    )

    display_data["고혈압 여부"] = (
        display_data["고혈압 여부"]
        .map({0: "아니요", 1: "예"})
        .fillna("확인 불가")
    )

    display_data["심장병 여부"] = (
        display_data["심장병 여부"]
        .map({0: "아니요", 1: "예"})
        .fillna("확인 불가")
    )

    return display_data


# 두 목록을 나란히 표시
left_col, right_col = st.columns(2)

with left_col:
    st.subheader(
        f"놓친 환자: {len(missed_patients):,}명"
    )

    st.write(
        "실제로 뇌졸중이지만 모델이 뇌졸중이 아니라고 예측한 사람입니다."
    )

    missed_display = prepare_person_list(missed_patients)

    st.dataframe(
        missed_display.head(20),
        use_container_width=True,
        hide_index=True
    )

    if len(missed_patients) > 20:
        st.caption(
            f"전체 {len(missed_patients):,}명 중 "
            "나이가 많은 순서로 앞의 20명만 표시했습니다."
        )

with right_col:
    st.subheader(
        f"헛짚은 사람: {len(false_alarm_patients):,}명"
    )

    st.write(
        "실제로 뇌졸중이 아니지만 모델이 뇌졸중이라고 예측한 사람입니다."
    )

    false_alarm_display = prepare_person_list(false_alarm_patients)

    st.dataframe(
        false_alarm_display.head(20),
        use_container_width=True,
        hide_index=True
    )

    if len(false_alarm_patients) > 20:
        st.caption(
            f"전체 {len(false_alarm_patients):,}명 중 "
            "나이가 많은 순서로 앞의 20명만 표시했습니다."
        )


# ==================================================
# 15. 목록과 혼동행렬의 값 비교
# ==================================================

st.subheader("목록과 혼동행렬 비교")

fn_count = len(missed_patients)
fp_count = len(false_alarm_patients)

st.write(
    f"선택한 모델의 놓친 환자 수는 **{fn_count:,}명**으로 "
    f"혼동행렬의 FN 값 **{selected_confusion['FN']:,}명**과 같습니다."
)

st.write(
    f"선택한 모델의 헛짚은 사람 수는 **{fp_count:,}명**으로 "
    f"혼동행렬의 FP 값 **{selected_confusion['FP']:,}명**과 같습니다."
)

if (
    fn_count == selected_confusion["FN"]
    and fp_count == selected_confusion["FP"]
):
    st.success(
        "두 목록의 인원수와 혼동행렬의 FN·FP 값이 일치합니다."
    )
else:
    st.error(
        "목록의 인원수와 혼동행렬의 값이 일치하지 않습니다."
    )


# ==================================================
# 16. 마무리 안내
# ==================================================

st.header("8. 결과 해석 안내")

st.write(
    "- 정확도: 전체 테스트 데이터 중 맞게 예측한 사람의 비율"
)

st.write(
    "- 재현율: 실제 뇌졸중인 사람 중 뇌졸중이라고 찾아낸 비율"
)

st.write(
    "- 정밀도: 뇌졸중이라고 예측한 사람 중 실제 뇌졸중인 비율"
)

st.write(
    "- F1: 정밀도와 재현율을 함께 고려한 지표"
)

st.warning(
    "이 프로그램은 분류 알고리즘을 학습하기 위한 교육용 실습입니다. "
    "실제 환자의 뇌졸중 여부를 진단하거나 의료적 판단을 내리는 용도로 "
    "사용할 수 없습니다."
)
