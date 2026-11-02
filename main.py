
import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/tkp12301/data/refs/heads/main/stroke.csv"

RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "뇌졸중_데이터_설명.json"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def load_records():
    if RECORD_FILE.exists():
        try:
            with open(RECORD_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("meanings", {}), data.get("source_note", "")
        except (OSError, json.JSONDecodeError):
            st.warning("기존 기록을 읽지 못했습니다.")
    return {}, ""


def save_records(meanings, source):
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "meanings": meanings,
        "source_note": source
    }

    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


st.title("🧠 뇌졸중 예측 실습실")
st.write("뇌졸중 관련 데이터를 분석하기 전 데이터의 구조와 특성을 살펴봅니다.")

try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.write("인터넷 연결과 CSV 주소를 확인해 주세요.")
    st.code(str(e))
    st.stop()


# 데이터 기본 통계
total_people = len(df)
total_columns = len(df.columns)

stroke_count = (
    (pd.to_numeric(df["stroke"], errors="coerce") == 1).sum()
    if "stroke" in df.columns else 0
)

stroke_ratio = (
    stroke_count / total_people * 100
    if total_people > 0 else 0
)


# 숫자 카드
st.subheader("📊 데이터 한눈에 보기")

c1, c2, c3, c4 = st.columns(4)

c1.metric("전체 사람 수", f"{total_people:,}명")
c2.metric("열 개수", f"{total_columns}개")
c3.metric("stroke가 1인 사람 수", f"{stroke_count:,}명")
c4.metric("stroke가 1인 비율", f"{stroke_ratio:.2f}%")


st.divider()


# 열 정보 입력
st.subheader("📋 열 이름과 데이터 특성")
st.write("교재를 참고하여 각 열의 우리말 뜻을 직접 작성하세요.")

saved_meanings, saved_source = load_records()

meanings = {}

for start in range(0, len(df.columns), 3):
    cols = st.columns(3)

    for i, name in enumerate(df.columns[start:start + 3]):
        with cols[i]:
            meanings[name] = st.text_input(
                f"{name}의 우리말 뜻",
                value=saved_meanings.get(name, ""),
                placeholder="우리말 뜻 입력",
                key=f"meaning_{name}"
            )


# 표 만들기
table_data = []

for name in df.columns:
    table_data.append({
        "열 이름": name,
        "우리말 뜻": meanings.get(name, ""),
        "값의 종류": str(df[name].dtype),
        "빈 값 개수": int(df[name].isna().sum())
    })

summary = pd.DataFrame(table_data)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)


st.divider()


# 처음 다섯 줄
st.subheader("🔍 데이터 첫 다섯 줄")

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)


st.divider()


# 출처 입력
st.subheader("📚 데이터 출처")

source_note = st.text_area(
    "교재에 나온 데이터 출처를 입력하세요.",
    value=saved_source,
    placeholder="출처 입력",
    height=100
)


# 저장 버튼
if st.button("💾 우리말 뜻과 출처 저장", type="primary"):
    try:
        save_records(meanings, source_note)
        st.success("입력한 내용이 성공적으로 저장되었습니다.")
    except OSError as e:
        st.error(f"저장 실패: {e}")


st.caption(f"데이터 출처 주소: {DATA_URL}")
st.caption("저장 위치: 내_기록/뇌졸중_데이터_설명.json")
