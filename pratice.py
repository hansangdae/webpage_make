# 라이브러리 불러오기
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy 
import statsmodels.api as sm

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
    ("csv 파일 분석", "여러가지 확률분포", '상관&회귀 분석') # 기능
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

# 기능 2. 확률 분포 설정
elif selected_page == "여러가지 확률분포":
    st.title("여러가지 확률분포")
    st.write("여러가지 확률분포를 시험해보세요.")

    # 이산, 연속 확률분포 선택
    col1, col2 = st.columns([1, 5])
    with col1:
        statistice_dist = st.selectbox(
        "확률 분포를 선택하세요:",
        ('이산 확률 분포', '연속 확률 분포', '표본 분포', '베이지안 관련 분포') # 기능
        )

    # 1) 이산 확률 분포 선택 시
    if statistice_dist == '이산 확률 분포':

        selected_dist = st.radio('원하는 이산 확률 분포를 선택하세요.', ['이항 분포', '포아송 분포'])

        st.markdown("---")

        # 1)-1 : 이산 확률 분포 - 이항 분포 선택 시
        
        if selected_dist == '이항 분포':
            st.subheader('이항 분포를 선택하셨습니다.')

            st.text('이항 분포 정의 : 성공할 확률이 p인 독립적인 시행을 n번 반복했을 때, 성공 횟수 X가 가질 수 있는 확률 분포')
            st.latex(r'''P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}''')

            st.text('원하는 모수를 입력하세요.')

            col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
            with col1:
                n = st.number_input('n을 입력하세요.', min_value = 1, key = 'dist_bi_n')
            
            with col2:
                p = st.number_input('p를 입력하세요.', min_value = 0.0000000001, max_value = 0.99999999999999, key = 'dist_bi_p')

            with col3:
                sample_n = st.number_input('원하는 샘플 개수를 입력하세요.', min_value = 1, key = 'dist_bi_samplen')

            with col4:
                st.write("")
                st.write("")
                anal_button = st.button('분석하기', key = 'dist_bi_button')

            st.markdown("---")

            if anal_button:
                col1, col2 = st.columns([1,1])
                with col1:
                    sample = scipy.stats.binom.rvs(n = n, p = p, size = sample_n)
                    sample_df = pd.DataFrame(sample, columns=['성공 횟수 (k)'])
                    st.subheader('샘플들의 기초통계량')
                    st.dataframe(sample_df.describe(), use_container_width=True)

                    st.subheader('샘플들의 히스토그램')
                    fig_bi, ax_bi = plt.subplots(figsize = (10, 10))
                    ax_bi.hist(sample)
                    st.pyplot(fig_bi)
                    plt.close(fig_bi)

        # 1)-2 : 이산 확률 분포 - 포아송 분포 선택 시
        if selected_dist == '포아송 분포':
            st.subheader('포아송 분포를 선택하셨습니다.')

            st.text('포아송 분포 정의 : 어떤 특정 시간 간격 또는 공간 영역 내에서 어떤 사건이 평균적으로 λ번 발생할 때, 해당 시간 간격 또는 공간 영역에서 그 사건이 k번 발생할 확률분포')
            st.latex(r'''P(X=k) = \frac{e^{-\lambda} \lambda^k}{k!} \quad \text{for } k \in \{0, 1, 2, \dots\}''')

            st.text('원하는 모수를 입력하세요.')

            col1, col2, col3, col4 = st.columns([1,1,1,3])
            with col1:
                n = st.number_input('n을 입력하세요.', min_value = 1, key = 'dist_poi_n')
            
            with col2:
                st.write("")
            with col3:
                sample_n = st.number_input('원하는 샘플 개수를 입력하세요.', min_value = 1, key = 'dist_poi_samplen')
            with col4:
                st.write("")
                st.write("")
                anal_button = st.button('분석하기', key = 'dist_poi_button')

            st.markdown("---")

            if anal_button:
                col1, col2 = st.columns([1,1])
                with col1:
                    sample = scipy.stats.poisson.rvs(mu = n, size = sample_n)
                    sample_df = pd.DataFrame(sample, columns=['발생 횟수 (k)'])
                    st.subheader('샘플들의 기초통계량')
                    st.dataframe(sample_df.describe(), use_container_width=True)

                    st.subheader('샘플들의 히스토그램')
                    fig_poi, ax_poi = plt.subplots(figsize = (10, 10))
                    ax_poi.hist(sample)
                    st.pyplot(fig_poi)
                    plt.close(fig_poi)

    # 2) 연속 확률 분포 선택 시
    if statistice_dist == '연속 확률 분포':

        selected_con = st.radio('원하는 연속 확률 분포를 선택하세요.', ['균등 분포', '정규 분포', '지수 분포'])

        st.markdown("---")

        # 2)-1 : 연속 확률 분포 - 균등 분포 선택 시
        
        if selected_con == '균등 분포':
            st.subheader('균등 분포를 선택하셨습니다.')

            st.text('균등 분포 정의 : 정해진 구간 내의 모든 값이 동일한 확률로 발생할 때 사용되는 확률 분포')
            st.latex(r'''P(X) = \begin{cases} \frac{1}{b-a} & \text{for } a \le x \le b \\ 0 & \text{otherwise} \end{cases}''')

            st.text('원하는 모수를 입력하세요.')

            col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
            with col1:
                a = st.number_input('최소값을 입력하세요.', key = 'dist_uni_a')
            
            with col2:
                b = st.number_input('최대값을 입력하세요.', key = 'dist_uni_b')

            with col3:
                sample_n = st.number_input('원하는 샘플 개수를 입력하세요.', min_value = 1, key = 'dist_uni_samplen')

            with col4:
                st.write("")
                st.write("")
                anal_button = st.button('분석하기', key = 'dist_uni_button')

            st.markdown("---")

            if anal_button:
                col1, col2 = st.columns([1,1])
                with col1:
                    sample = scipy.stats.uniform.rvs(loc = a, scale = b-a, size = sample_n)
                    sample_df = pd.DataFrame(sample, columns=['통계량'])
                    st.subheader('샘플들의 기초통계량')
                    st.dataframe(sample_df.describe(), use_container_width=True)

                    st.subheader('샘플들의 히스토그램 & PDF')
                    fig_uni, ax_uni = plt.subplots(figsize = (10, 10))
                    ax_uni.hist(sample, density=True, label = 'sample')

                    x_pdf_plot = np.linspace(a - (b - a) * 0.2, b + (b - a) * 0.2, 500)
                    uni_pdf = scipy.stats.uniform.pdf(x_pdf_plot, loc=a, scale=b-a)
                    ax_uni.plot(x_pdf_plot, uni_pdf, color = 'yellow', label = 'PDF')
                    ax_uni.fill_between(x_pdf_plot, 0, uni_pdf, color='yellow', alpha=0.3)

                    fig_uni.legend()
                    st.pyplot(fig_uni)
                    plt.close(fig_uni)

        # 2)-2 : 연속 확률 분포 - 정규 분포 선택 시
        
        if selected_con == '정규 분포':
            st.subheader('정규 분포를 선택하셨습니다.')

            st.text('정규 분포 정의 : 평균을 중심으로 좌우 대칭인 종 모양을 띠는 연속 확률 분포')
            st.latex(r'''P(X=x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}''')

            st.text('원하는 모수를 입력하세요.')

            col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
            with col1:
                mu = st.number_input('평균을 입력하세요.', key = 'dist_stan_mu')
            
            with col2:
                std = st.number_input('표준편차를 입력하세요.', key = 'dist_stan_std')

            with col3:
                sample_n = st.number_input('원하는 샘플 개수를 입력하세요.', min_value = 1, key = 'dist_stan_samplen')

            with col4:
                st.write("")
                st.write("")
                anal_button = st.button('분석하기', key = 'dist_stan_button')

            st.markdown("---")

            if anal_button:
                col1, col2 = st.columns([1,1])
                with col1:
                    sample = scipy.stats.norm.rvs(loc = mu, scale = std, size = sample_n)
                    sample_df = pd.DataFrame(sample, columns=['통계량'])
                    st.subheader('샘플들의 기초통계량')
                    st.dataframe(sample_df.describe(), use_container_width=True)

                    st.subheader('샘플들의 히스토그램 & PDF')
                    fig_stan, ax_stan = plt.subplots(figsize = (10, 10))
                    ax_stan.hist(sample, density=True, label = 'sample')

                    x_pdf_plot = np.linspace(mu-4*std, mu+4*std, 500)
                    stan_pdf = scipy.stats.norm.pdf(x_pdf_plot, loc=mu, scale=std)
                    ax_stan.plot(x_pdf_plot, stan_pdf, color = 'yellow', label = 'PDF')
                    ax_stan.fill_between(x_pdf_plot, 0, stan_pdf, color='yellow', alpha=0.3)

                    fig_stan.legend()
                    st.pyplot(fig_stan)
                    plt.close(fig_stan)

        # 2)-3 : 연속 확률 분포 - 지수 분포 선택 시
        
        if selected_con == '지수 분포':
            st.subheader('지수 분포를 선택하셨습니다.')

            st.text('지수 분포 정의 : 포아송 분포(평균 = 람다)를 따르는 과정에서 다음 사건이 발생할 때까지 걸리는 시간을 모델링하는 확률 분포')
            st.latex(r'''P(X=x) = \lambda e^{-\lambda x} \quad \text{for } x \ge 0''')

            st.text('원하는 모수를 입력하세요.')

            col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
            with col1:
                lamb = st.number_input('람다를 입력하세요.', min_value = 0.0000001, key = 'dist_exp_lambda')
            
            with col2:
                st.write("")

            with col3:
                sample_n = st.number_input('원하는 샘플 개수를 입력하세요.', min_value = 1, key = 'dist_exp_samplen')

            with col4:
                st.write("")
                st.write("")
                anal_button = st.button('분석하기', key = 'dist_lambda_button')

            st.markdown("---")

            if anal_button:
                col1, col2 = st.columns([1,1])
                with col1:
                    sample = scipy.stats.expon.rvs(loc = 0, scale = 1/lamb, size = sample_n)
                    sample_df = pd.DataFrame(sample, columns=['통계량'])
                    st.subheader('샘플들의 기초통계량')
                    st.dataframe(sample_df.describe(), use_container_width=True)

                    st.subheader('샘플들의 히스토그램 & PDF')
                    fig_exp, ax_exp = plt.subplots(figsize = (10, 10))
                    ax_exp.hist(sample, density=True, label = 'sample')

                    x_pdf_plot = np.linspace(0, (1/lamb)*10, 500)
                    exp_pdf = scipy.stats.expon.pdf(x_pdf_plot, loc=0, scale=1/lamb)
                    ax_exp.plot(x_pdf_plot, exp_pdf, color = 'yellow', label = 'PDF')
                    ax_exp.fill_between(x_pdf_plot, 0, exp_pdf, color='yellow', alpha=0.3)

                    fig_exp.legend()
                    st.pyplot(fig_exp)
                    plt.close(fig_exp)

    # 3) 표본 분포 선택 시
    if statistice_dist == '표본 분포':

        selected_sam = st.radio('원하는 표본 분포를 선택하세요.', ['t-분포', '카이제곱 분포', 'F-분포'])

        st.markdown("---")

        # 3)-1 : 표본 분포 - t-분포 선택 시
        if selected_sam == 't-분포':
            st.subheader('t-분포를 선택하셨습니다.')

            st.text('t-분포 정의 : 표본의 크기가 작고 모집단의 표준편차를 알 수 없을 때 모평균의 추정이나 가설검정에 사용되는 확률분포')
            st.text('주요 특징 : 1) 종모양과 대칭성 2) 모수는 자유도 1개를 가짐 3) 정규분포 대비 꼬리가 두꺼움(모집단의 표준편차를 사용하지 않음에 대한 불확실성을 의미)')
            st.text('           '+'4) 자유도가 커질수록 정규분포에 가까워짐-일반적으로 30이상이면 정규분포로 고려')

            col1, col2 = st.columns([1,1])
            with col1:
                st.text('통계량 공식')
                st.latex(r'''t = \frac{\bar{X} - \mu}{s/\sqrt{n}}''')

            with col2:
                st.text('확률밀도함수')
                st.latex(r'''f(t) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\Gamma\left(\frac{\nu}{2}\right)}\left(1 + \frac{t^2}{\nu}\right)^{-\frac{\nu+1}{2}}''')
            
            st.markdown("---")

            st.text('원하는 모수를 입력하세요.')

            col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
            with col1:
                v = st.number_input('자유도를 입력하세요.', min_value = 0.0000001, key = 'dist_t_dof')
            
            with col2:
                st.write("")

            with col3:
                sample_n = st.number_input('원하는 샘플 개수를 입력하세요.', min_value = 1, key = 'dist_t_samplen')

            with col4:
                st.write("")
                st.write("")
                anal_button = st.button('분석하기', key = 'dist_t_button')

            st.markdown("---")

            if anal_button:
                col1, col2 = st.columns([1,1])
                with col1:
                    sample = scipy.stats.t.rvs(df = v, loc = 0, scale = 1, size = sample_n)
                    sample_df = pd.DataFrame(sample, columns=['통계량'])
                    st.subheader('샘플들의 기초통계량')
                    st.dataframe(sample_df.describe(), use_container_width=True)

                    st.subheader('샘플들의 히스토그램 & PDF')
                    fig_t, ax_t = plt.subplots(figsize = (10, 10))
                    ax_t.hist(sample, density=True, label = 'sample')

                    x_pdf_plot = np.linspace(-5, 5, 500)
                    t_pdf = scipy.stats.t.pdf(x_pdf_plot, df = v, loc=0, scale=1)
                    ax_t.plot(x_pdf_plot, t_pdf, color = 'yellow', label = 'PDF')
                    ax_t.fill_between(x_pdf_plot, 0, t_pdf, color='yellow', alpha=0.3)

                    fig_t.legend()
                    st.pyplot(fig_t)
                    plt.close(fig_t)

        # 3)-2 : 표본 분포 - 카이제곱분포 선택 시
        if selected_sam == '카이제곱 분포':
            st.subheader('카이제곱 분포를 선택하셨습니다.')

            st.text('카이제곱 분포 정의 : k개의 서로 독립적인 표준정규분포를 따르는 확률분포들을 각각 제곱하여 합한 값을 자유도 k인 카이제곱 분포이라 함')
            st.text('분산에 대한 추론이나 범주형 데이터의 분석에 주로 쓰임')
            st.text('주요 특징 : 1) 0이상의 값을 가지며 비대칭 2) 모수는 자유도 1개를 가짐 3) 가법성(서로 독립인 두 카이제곱 분포를 더할 시, 결과도 카이제곱 분포를 따르며 자유도는 두 분포 자유도를 더한 값)')

            col1, col2 = st.columns([1,1])
            with col1:
                st.text('통계량 공식')
                st.latex(r'''\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}''')

            with col2:
                st.text('확률밀도함수')
                st.latex(r'''f(x; k) = \frac{1}{2^{k/2} \Gamma(k/2)} x^{k/2 - 1} e^{-x/2} \quad \text{for } x > 0''')
            
            st.markdown("---")

            st.text('원하는 모수를 입력하세요.')

            col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
            with col1:
                v = st.number_input('자유도를 입력하세요.', min_value = 0.0000001, key = 'dist_chi_dof')
            
            with col2:
                st.write("")

            with col3:
                sample_n = st.number_input('원하는 샘플 개수를 입력하세요.', min_value = 1, key = 'dist_chi_samplen')

            with col4:
                st.write("")
                st.write("")
                anal_button = st.button('분석하기', key = 'dist_chi_button')

            st.markdown("---")

            if anal_button:
                col1, col2 = st.columns([1,1])
                with col1:
                    sample = scipy.stats.chi2.rvs(df = v, loc = 0, scale = 1, size = sample_n)
                    sample_df = pd.DataFrame(sample, columns=['통계량'])
                    st.subheader('샘플들의 기초통계량')
                    st.dataframe(sample_df.describe(), use_container_width=True)

                    st.subheader('샘플들의 히스토그램 & PDF')
                    fig_chi, ax_chi = plt.subplots(figsize = (10, 10))
                    ax_chi.hist(sample, density=True, label = 'sample')

                    x_pdf_plot = np.linspace(0, v+4*np.sqrt(2*v), 500)
                    chi_pdf = scipy.stats.chi2.pdf(x_pdf_plot, df = v, loc=0, scale=1)
                    ax_chi.plot(x_pdf_plot, chi_pdf, color = 'yellow', label = 'PDF')
                    ax_chi.fill_between(x_pdf_plot, 0, chi_pdf, color='yellow', alpha=0.3)

                    fig_chi.legend()
                    st.pyplot(fig_chi)
                    plt.close(fig_chi)

        # 3)-3 : 표본 분포 - F-분포 선택 시
        if selected_sam == 'F-분포':
            st.subheader('F-분포를 선택하셨습니다.')

            st.text('F-분포 정의 : 2개의 독립적인 카이제곱 분포를 따르는 확률변수를 각각의 자유로나 나눈 비율이 따르는 확률분포')
            st.text('집단간의 분산을 비교하는데 쓰임')
            st.text('주요 특징 : 1) 0이상의 값을 가지며 비대칭 2) 모수는 2개를 가짐(분자/분모의 자유도) 3) 자유도가 n인 t-분포를 제곱하며 (1,n)의 F-분포와 같음')

            col1, col2 = st.columns([1,1])
            with col1:
                st.text('통계량 공식')
                st.latex(r'''F = \frac{U/d_1}{V/d_2}''')

            with col2:
                st.text('확률밀도함수')
                st.latex(r'''f(x; d_1, d_2) = \frac{\sqrt{\frac{(d_1 x)^{d_1} d_2^{d_2}}{(d_1 x + d_2)^{d_1+d_2}}}}{x \text{B}\left(\frac{d_1}{2}, \frac{d_2}{2}\right)} \quad \text{for } x > 0''')
            
            st.markdown("---")

            st.text('원하는 모수를 입력하세요.')

            col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
            with col1:
                v1 = st.number_input('자유도1(분자)를 입력하세요.', min_value = 0.0000001, key = 'dist_F_dof1')
            
            with col2:
                v2 = st.number_input('자유도2(분모)를 입력하세요.', min_value = 0.0000001, key = 'dist_F_dof2')

            with col3:
                sample_n = st.number_input('원하는 샘플 개수를 입력하세요.', min_value = 1, key = 'dist_F_samplen')

            with col4:
                st.write("")
                st.write("")
                anal_button = st.button('분석하기', key = 'dist_F_button')

            st.markdown("---")

            if anal_button:
                col1, col2 = st.columns([1,1])
                with col1:
                    sample = scipy.stats.f.rvs(dfn = v1, dfd = v2, loc = 0, scale = 1, size = sample_n)
                    sample_df = pd.DataFrame(sample, columns=['통계량'])
                    st.subheader('샘플들의 기초통계량')
                    st.dataframe(sample_df.describe(), use_container_width=True)

                    st.subheader('샘플들의 히스토그램 & PDF')
                    fig_f, ax_f = plt.subplots(figsize = (10, 10))
                    ax_f.hist(sample, density=True, label = 'sample')

                    x_max_pdf = scipy.stats.f.ppf(0.999, dfn=v1, dfd=v2)

                    if x_max_pdf < 5:
                        x_max_pdf = 5

                    x_pdf_plot = np.linspace(0, x_max_pdf*1.2, 500)
                    f_pdf = scipy.stats.f.pdf(x_pdf_plot, dfn = v1, dfd = v2, loc=0, scale=1)
                    ax_f.plot(x_pdf_plot, f_pdf, color = 'yellow', label = 'PDF')
                    ax_f.fill_between(x_pdf_plot, 0, f_pdf, color='yellow', alpha=0.3)

                    fig_f.legend()
                    st.pyplot(fig_f)
                    plt.close(fig_f)

    # 4) 베이지안 관련 분포 선택 시
    if statistice_dist == '베이지안 관련 분포':

        selected_bae = st.radio('원하는 베이지안 관련 분포를 선택하세요.', ['감마 분포', '베타 분포', '역감마 분포'])

        st.markdown("---")

        # 4)-1 : 베이지안 관련 분포 - 감마 분포 선택 시    
        if selected_bae == '감마 분포':
            st.subheader('감마 분포를 선택하셨습니다.')

            st.text('감마 분포 정의 : 평균적으로 b만큼의 시간이 걸리는 사건이 a번 발생할때 까지 걸리는 총 시간이 따르는 확률분포')
            st.text('참고 : 지수 분포(평균적으로 b만큼의 시간이 걸리는 사건이 처음 발생할때 까지 걸리는 시간이 따르는 확률분포)')
            st.text('지수 분포의 일반화된 형태로 볼 수 있음')
            st.text('a(형태 모수) : 분포의 모양을 결정, b(척도 모수) : 분포의 스케일, 퍼짐을 결정')
            st.text('포아송/지수 분포의 모수, 정규 분포의 정밀도(분산의 역수)에 대한 켤레 사전 분포로 쓰임')

            col1, col2 = st.columns([1,1])
            with col1:
                st.text('감마함수')
                st.latex(r'''\Gamma(z) = \int_0^\infty t^{z-1} e^{-t} dt''')

            with col2:
                st.text('확률밀도함수')
                st.latex(r'''f(x; \alpha, \beta) = \frac{1}{\Gamma(\alpha)\beta^\alpha} x^{\alpha-1} e^{-x/\beta} \quad \text{for } x > 0''')
            
            st.markdown("---")

            st.text('원하는 모수를 입력하세요.')

            col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
            with col1:
                a = st.number_input('형태 모수(사건 발생 횟수)를 입력하세요.', min_value = 0.0000001, key = 'dist_gamma_a')
            
            with col2:
                b = st.number_input('척도 모수(평균 발생 시간)를 입력하세요.', min_value = 0.0000001, key = 'dist_gamma_b')

            with col3:
                sample_n = st.number_input('원하는 샘플 개수를 입력하세요.', min_value = 1, key = 'dist_gamma_samplen')

            with col4:
                st.write("")
                st.write("")
                anal_button = st.button('분석하기', key = 'dist_gamma_button')

            st.markdown("---")

            if anal_button:
                col1, col2 = st.columns([1,1])
                with col1:
                    sample = scipy.stats.gamma.rvs(a = a, scale = b, size = sample_n)
                    sample_df = pd.DataFrame(sample, columns=['통계량'])
                    st.subheader('샘플들의 기초통계량')
                    st.dataframe(sample_df.describe(), use_container_width=True)

                    st.subheader('샘플들의 히스토그램 & PDF')
                    fig_gamma, ax_gamma = plt.subplots(figsize = (10, 10))
                    ax_gamma.hist(sample, density=True, label = 'sample')

                    expected_max_x = a * b + 4 * b * np.sqrt(a)
                    plot_max_x_hist = max(np.max(sample) * 1.1, expected_max_x * 1.2)
        
                    if plot_max_x_hist < 5: 
                        plot_max_x_hist = 5

                    x_pdf_plot = np.linspace(0, plot_max_x_hist*1.05, 500)
                    gamma_pdf = scipy.stats.gamma.pdf(x_pdf_plot, a = a, scale=b)
                    ax_gamma.plot(x_pdf_plot, gamma_pdf, color = 'yellow', label = 'PDF')
                    ax_gamma.fill_between(x_pdf_plot, 0, gamma_pdf, color='yellow', alpha=0.3)

                    fig_gamma.legend()
                    st.pyplot(fig_gamma)
                    plt.close(fig_gamma)

        # 4)-2 : 베이지안 관련 분포 - 베타 분포 선택 시    
        if selected_bae == '베타 분포':
            st.subheader('베타 분포를 선택하셨습니다.')

            st.text('베타 분포 정의 : 총 시행횟수(n) 중, 성공확률이 가지는 확률분포')
            st.text('a : 성공 확률이 1쪽에 가까워지는 형태 모수, b : 성공 확률이 0쪽에 가까워지는 형태 모수')
            st.text('이항 분포의 성공확률에 대한 켤레 사전 분포로 쓰임')

            col1, col2 = st.columns([1,1])
            with col1:
                st.text('베타함수')
                st.latex(r'''B(\alpha, \beta) = \int_0^1 t^{\alpha-1}(1-t)^{\beta-1} dt''')

            with col2:
                st.text('확률밀도함수')
                st.latex(r'''f(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)} \quad \text{for } 0 < x < 1''')
            
            st.markdown("---")

            st.text('원하는 모수를 입력하세요.')

            col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
            with col1:
                a = st.number_input('형태 모수(1쪽)를 입력하세요.', min_value = 0.0000001, key = 'dist_beta_a')
            
            with col2:
                b = st.number_input('형태 모수(0쪽)를 입력하세요.', min_value = 0.0000001, key = 'dist_beta_b')

            with col3:
                sample_n = st.number_input('원하는 샘플 개수를 입력하세요.', min_value = 1, key = 'dist_beta_samplen')

            with col4:
                st.write("")
                st.write("")
                anal_button = st.button('분석하기', key = 'dist_beta_button')

            st.markdown("---")

            if anal_button:
                col1, col2 = st.columns([1,1])
                with col1:
                    sample = scipy.stats.beta.rvs(a, b, loc = 0, scale = 1, size = sample_n)
                    sample_df = pd.DataFrame(sample, columns=['통계량'])
                    st.subheader('샘플들의 기초통계량')
                    st.dataframe(sample_df.describe(), use_container_width=True)

                    st.subheader('샘플들의 히스토그램 & PDF')
                    fig_beta, ax_beta = plt.subplots(figsize = (10, 10))
                    ax_beta.hist(sample, density=True, label = 'sample')

                    x_pdf_plot = np.linspace(0, 1, 500)
                    beta_pdf = scipy.stats.beta.pdf(x_pdf_plot, a = a, b = b, loc = 0, scale=1)
                    ax_beta.plot(x_pdf_plot, beta_pdf, color = 'yellow', label = 'PDF')
                    ax_beta.fill_between(x_pdf_plot, 0, beta_pdf, color='yellow', alpha=0.3)

                    fig_beta.legend()
                    st.pyplot(fig_beta)
                    plt.close(fig_beta)

        # 4)-2 : 베이지안 관련 분포 - 역감마 분포 선택 시    
        if selected_bae == '역감마 분포':
            st.subheader('역감마 분포를 선택하셨습니다.')

            st.text('역감마 분포 정의 : 감마 분포의 역함수 형태의 확률분포. 분산 모델링에 쓰임')
            st.text('a(형태 모수) : 분포의 모양을 결정, b(척도 모수) : 분포의 스케일, 퍼짐을 결정')
            st.text('정규 분포의 분산에 대한 켤레 사전 분포로 쓰임')

            col1, col2 = st.columns([1,1])
            with col1:
                st.text('확률밀도함수')
                st.latex(r'''f(x; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{-\alpha-1} e^{-\beta/x} \quad \text{for } x > 0''')

            with col2:
                st.write("")
            
            st.markdown("---")

            st.text('원하는 모수를 입력하세요.')

            col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
            with col1:
                a = st.number_input('형태 모수를 입력하세요.', min_value = 0.0000001, key = 'dist_inverse_gamma_a')
            
            with col2:
                b = st.number_input('척도 모수를 입력하세요.', min_value = 0.0000001, key = 'dist_inverse_gamma_b')

            with col3:
                sample_n = st.number_input('원하는 샘플 개수를 입력하세요.', min_value = 1, key = 'dist_inverse_gamma_samplen')

            with col4:
                st.write("")
                st.write("")
                anal_button = st.button('분석하기', key = 'dist_inverse_gamma_button')

            st.markdown("---")

            if anal_button:
                col1, col2 = st.columns([1,1])
                with col1:
                    sample = scipy.stats.invgamma.rvs(a = a, scale = b, size = sample_n)
                    sample_df = pd.DataFrame(sample, columns=['통계량'])
                    st.subheader('샘플들의 기초통계량')
                    st.dataframe(sample_df.describe(), use_container_width=True)

                    st.subheader('샘플들의 히스토그램 & PDF')
                    fig_inv_gamma, ax_inv_gamma = plt.subplots(figsize = (10, 10))
                    ax_inv_gamma.hist(sample, density=True, label = 'sample')

                    plot_max_x_hist = max(np.max(sample) * 1.1, scipy.stats.invgamma.ppf(0.999, a=a, scale=b) * 1.2)
                    if plot_max_x_hist < 5: 
                        plot_max_x_hist = 5
                        
                    x_pdf_plot = np.linspace(0, plot_max_x_hist*1.05, 500)
                    inv_gamma_pdf = scipy.stats.invgamma.pdf(x_pdf_plot, a = a, scale=b)
                    ax_inv_gamma.plot(x_pdf_plot, inv_gamma_pdf, color = 'yellow', label = 'PDF')
                    ax_inv_gamma.fill_between(x_pdf_plot, 0, inv_gamma_pdf, color='yellow', alpha=0.3)

                    fig_inv_gamma.legend()
                    st.pyplot(fig_inv_gamma)
                    plt.close(fig_inv_gamma)

# 기능 3. 상관&회귀 분석
if selected_page == "상관&회귀 분석":
    st.title("상관&회귀 분석 페이지")
    st.write("csv 파일을 올리세요. 그리고 상관 & 회귀 분석을 해보세요.")
    
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

        # 상관 or 회귀 분석 선택
        selected_cor_reg = st.radio('원하는 분석을 선택하세요.', ['상관 분석', '회귀 분석'])

        # 상관 분석 선택 시
        if selected_cor_reg == '상관 분석':

            # 분석 대상 컬럼명 선택
            st.markdown("---")

            st.text('분석하고자 하는 수치형 데이터 컬럼명 2개를 선택하세요')
            numeric_columns = df.select_dtypes(include=np.number).columns

            col1, col2, col3, col4 = st.columns([1,1,1,3])
            with col1:
                selected_column1 = st.selectbox("컬럼명 1:", numeric_columns)
            
            numeric_columns_copy = numeric_columns 
            with col2:
                selected_column2 = st.selectbox("컬럼명 2:", numeric_columns_copy.drop(selected_column1))

            with col3:
                st.write("")
                st.write("")
                anal_button = st.button('분석하기', key = 'cor_button')

            with col4:
                st.write("")

            st.markdown("---")

            # 상관계수 계산
            new_df = df.dropna(subset = [selected_column1, selected_column2])

            if anal_button:
                st.subheader('상관분석을 진행합니다.')
                corr_coef, _ = scipy.stats.pearsonr(new_df[selected_column1].values, new_df[selected_column2].values)
                st.text('상관계수는 {0}입니다.'.format(np.round(corr_coef, 3)))

                # 산점도 
                st.markdown("---")
                st.subheader('산점도')
                fig_scatter, ax_scatter = plt.subplots(figsize = (10,8))
                ax_scatter.scatter(new_df[selected_column1], new_df[selected_column2])
                st.pyplot(fig_scatter)
                plt.title('산점도')
                plt.close(fig_scatter)

        # 회귀 분석 선택 시
        if selected_cor_reg == '회귀 분석':

            # 분석 독립변수 컬럼명 선택
            st.markdown("---")

            st.subheader('변수 선택')

            col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

            with col1:
                ind_num = st.number_input("분석하고자 하는 독립변수 개수를 선택하세요:", min_value=1, key='ind_num')

                selected_columns = [] 

                # 데이터프레임의 실제 컬럼명 목록 (selectbox 옵션으로 사용)
                available_columns = df.select_dtypes(include=np.number).columns.tolist()

                for i in range(ind_num):
                    selected_col_ind = st.selectbox(f"{i+1}번째 컬럼을 선택하세요:", available_columns, key=f"column_select_{i}")
                    selected_columns.append(selected_col_ind)

            with col2:
                # 분석 종속변수 컬럼명 선택
                st.write("")
                selected_col_res = st.selectbox('종속 변수 컬럼을 선택하세요:', available_columns, key = 'selected_col_res_name')

            with col3:
                st.write("")
                st.write("")
                anal_button = st.button('분석하기', key = 'reg_button')

            with col4:
                st.write("")

            # 선택된 컬럼들 확인 
            st.markdown("---")
            st.subheader('회귀 분석')

            if anal_button:
                col1, col2 = st.columns([1,1])

                with col1:
                    st.text('선택된 독립 변수 컬럼들 : {0}'.format(selected_columns))

                with col2:
                    st.text('선택된 종속 변수 컬럼 : {0}'.format(selected_col_res))

            # 회귀 분석 1 : 독립변수가 1개일때(그래프도 표현)
            columns_sum = selected_columns + [selected_col_res]
            new_df = df.dropna(subset = columns_sum)

            if len(selected_columns) == 1:

                # 회귀식&설명계수&회귀계수 유의확률
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(new_df[selected_col_ind], new_df[selected_col_res])
                regression_equation = f"Y = {slope:.2f}X + {intercept:.2f}"
                reg_data = {'회귀식': [regression_equation], '설명계수(R^2)': [np.round(r_value**2,5)], '회귀계수 유의확률(p_vaule)': [np.round(p_value, 10)]}

                reg_frame = pd.DataFrame(reg_data)
                st.dataframe(reg_frame, hide_index = True)

                # 회귀 곡선
                fig_reg, ax_reg = plt.subplots(figsize = (10,8))
                sns.regplot(x = selected_col_ind, y = selected_col_res, data = new_df, ax = ax_reg, line_kws={'color':'red', 'linestyle':'--'})
                st.pyplot(fig_reg)
                plt.title('회귀 직선 그래프')
                plt.close(fig_reg)

            # 회귀 분석 2 : 독립변수가 2개 이상일때(그래프 표현 X)
            if len(selected_columns) > 1:

                # OLS (Ordinary Least Squares) 모델 생성 및 학습(위의 scipy.stats.linregress 코드는 설명계수가 1개인 경우에만 유효)
                x_data = sm.add_constant(new_df[selected_columns])
                y_data = new_df[selected_col_res]
                model = sm.OLS(y_data, x_data)
                results = model.fit()

                # 회귀식 만들기
                coefficients = results.params
                intercept = coefficients['const']
                feature_coefficients = coefficients.drop('const').round(4)

                equation = f"Y = {intercept:.4f}"

                # 모든 독립 변수의 계수를 순회하며 회귀식에 추가
                # feature_coefficients에는 'X1', 'X2', 'X3' (그리고 다른 독립 변수가 있다면 모두)의 계수가 있습니다.
                for feature, coef in feature_coefficients.items():
                    if coef >= 0:
                        equation += f" + {coef:.4f} * {feature}"
                    else: # 계수가 음수일 경우 '+' 대신 '-'로 표시하고 절대값 사용
                        equation += f" - {abs(coef):.4f} * {feature}"


                st.text('회귀식 : ' + equation)

                st.code(results.summary().as_text())
