# 라이브러리 불러오기
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 웹 페이지 이름 만들기
st.set_page_config(
    page_title="간단한 통계 계산",
    layout="wide", 
#    page_icon =         계산기 이미지 추가해보기
)

# select box를 이용한 사이드바 생성(제목, 기능)
st.sidebar.header("기능 선택")

selected_page = st.sidebar.selectbox(
    "기능을 선택하세요:",
    ("csv 파일 분석", "데이터 분석", "시각화", "설정") # 기능
)

# 선택된 기능에 따라 메인 콘텐츠 변경

# 기능 1. csv 통계 분석
if selected_page == "csv 파일 분석":
    st.title("csv 파일 분석 페이지")
    st.write("csv 파일을 올리세요. 그리고 간단한 통계 계산을 해보세요.")
    
    # csv 파일 올리기
    uploaded_file = st.file_uploader(
    "파일을 선택하세요 (CSV)",
    type=["csv"] # 허용할 파일 확장자 지정 (일단 csv 파일만 선택)
    )

    st.markdown("---")

    # 파일 기본 정보 표현
    if uploaded_file is not None:
        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "filesize(byte)": uploaded_file.size}
        st.write("업로드된 파일 정보:")
        st.json(file_details) # 파일 정보를 JSON 형식으로 표시

        st.markdown("---")

        # 업로드된 csv 파일 처음 5행 출력
        if uploaded_file.type == "text/csv":
            st.subheader("CSV 파일 내용 미리보기 :")
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")

        st.markdown("---")

        # 분석 대상 컬럼명 선택
        st.text('분석하고자 하는 컬럼명을 선택하세요')
        column_list = df.columns
        col1, col2 = st.columns([1,5])
        with col1:
            selected_column = st.selectbox("컬럼명:", column_list)

        st.text('수치 데이터의 경우, unique한 값이 10개 이하이면 범주형 데이터로 취급합니다.')

        # 입력받은 컬럼이 '양적'인지 '질적'인지 확인(질적 : 데이터가 숫자&유니크한 값이 10개 이상인 경우, 양적 : 그 외 모두)
        col_contents_text = True
        if np.issubdtype(df[selected_column].dtype, np.number) & (len(df[selected_column].unique()) > 10):
            col_contents_text = False

        # 선택된 컬럼의 null값 제외된 데이터 프레임 새로 생성
        new_df = df.dropna(subset = [selected_column])
        
        st.markdown("---")
        
        # 질적 데이터인 경우, 막대그래프 or 원그래프 표현
        if col_contents_text == True:
            col1, col2 = st.columns([1,1])
            with col1:
                st.subheader('막대그래프')
                fig_bar, ax_bar = plt.subplots(figsize = (10, 8))
                unique = new_df[selected_column].unique()
                counts = new_df[selected_column].value_counts()

                cmap = plt.cm.get_cmap('viridis') 
                bar_colors = [cmap(i) for i in np.linspace(0, 1, len(unique))]

                ax_bar.bar(unique, counts, color = bar_colors)
                ax_bar.set_xticks(unique)
                st.pyplot(fig_bar)
                plt.close(fig_bar)

            with col2:
                st.subheader('원그래프')
                fig_circle, ax_circle = plt.subplots(figsize = (10, 10))
                ax_circle.pie(counts, labels = unique)
                st.pyplot(fig_circle)
                plt.close(fig_circle)

            # 추가 질적 데이터를 받아 해당 데이터 별(hue) 막대그래프 추가
            st.markdown("---")
            st.text('추가 구분하고자 하는 컬럼명을 선택하세요')
            col1, col2 = st.columns([1,5])
            with col1:
                column_list_copy = column_list
                selected_bar_column = st.selectbox("컬럼명:", column_list_copy.drop(selected_column))

            col1, col2 = st.columns([1,1])
            with col1:
                st.subheader('추가 구분 막대그래프')
                fig_bar_sns, ax_bar_sns = plt.subplots(figsize = (10, 8))

                sns.countplot(x = selected_column, data = new_df, hue = selected_bar_column, ax = ax_bar_sns)
                ax_bar_sns.set_xticks(unique)
                st.pyplot(fig_bar_sns)
                plt.close(fig_bar_sns)
        
        # 양적 데이터인 경우, 1) 기초 통계량, 2) 히스토그램&상자그림, 3) 다른 수치형 데이터와의 산점도
        # 1) 기초 통계량
        if col_contents_text == False:
            mean = new_df[selected_column].mean()
            min = new_df[selected_column].min()
            max = new_df[selected_column].max()
            std = new_df[selected_column].std()
            median = new_df[selected_column].median()

            st.subheader('기초 통계량')
            statistice_data = {'평균':[mean], '최소값':[min], '최대값':[max], '중앙값':[median], '표준편차':[std]}
            statistice_df = pd.DataFrame(statistice_data)
            st.dataframe(statistice_df,hide_index = True)

            # 2) 히스토그램 & 상자그림
            st.markdown("---")
            st.subheader('히스토그램')

            col1, col2 = st.columns([1,1])
            with col1:
                fig_hist, ax_hist = plt.subplots(figsize = (10,8))
                ax_hist.hist(new_df[selected_column])
                st.pyplot(fig_hist)
                plt.close(fig_hist)

            with col2:
                fig_box, ax_box = plt.subplots(figsize = (10,8))
                ax_box.boxplot(new_df[selected_column])
                st.pyplot(fig_box)
                plt.close(fig_box)

            # 3) 산점도
            st.markdown("---")

            # 같이 산점도를 그리고자 하는 대상 컬럼명 선택
            st.text('산점도를 그리고자 하는 대상 컬럼명을 선택하세요')
            col1, col2 = st.columns([1,5])
            with col1:
                column_list_copy = column_list
                selected_scattered_column = st.selectbox("컬럼명:", column_list_copy.drop(selected_column))

            st.subheader('산점도')
            fig_scatter, ax_scatter = plt.subplots(figsize = (10,8))
            ax_scatter.scatter(new_df[selected_column], new_df[selected_scattered_column])
            st.pyplot(fig_scatter)
            plt.close(fig_scatter)

    else:
        st.info("파일을 업로드하려면 '파일 찾아보기' 버튼을 클릭하거나 파일을 끌어다 놓으세요.")

    st.markdown("---")

elif selected_page == "데이터 분석":
    st.title("데이터 분석 페이지")
    st.write("여기는 데이터를 분석하는 기능이 들어갈 곳입니다.")
    st.dataframe({"A": [1, 2, 3], "B": [4, 5, 6]})
    st.button("데이터 새로고침")

elif selected_page == "시각화":
    st.title("시각화 페이지")
    st.write("다양한 차트와 그래프를 여기에 표시할 수 있습니다.")
    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame({
        "fruit": ["Apple", "Orange", "Banana", "Grape"],
        "amount": [4, 1, 2, 5],
        "city": ["New York", "Paris", "London", "Berlin"]
    })
    fig = px.bar(df, x="fruit", y="amount", color="city")
    st.plotly_chart(fig)

elif selected_page == "설정":
    st.title("설정 페이지")
    st.write("앱의 설정을 변경하는 옵션들을 여기에 배치할 수 있습니다.")
    st.slider("볼륨 조절", 0, 100, 50)
    st.checkbox("알림 받기")
