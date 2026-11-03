# 2차원 확률 데이터에서 최솟값과 최댓값 계산
probability_min = min(
    min(row) for row in logistic_probability_grid
)

probability_max = max(
    max(row) for row in logistic_probability_grid
)

# 확률 0.5 결정 경계가 그래프 범위 안에 있는지 확인
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
