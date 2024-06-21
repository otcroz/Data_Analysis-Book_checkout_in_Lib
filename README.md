# 서울시 도서관 대출 현황 및 기상 요소를 활용한 도서관 대출자 수 예측 모델

* Related: 2024-1 빅데이터 기말 프로젝트 <br />
* 진행기간: 2024.6.2 ~ 2024.6.9

![슬라이드2](https://github.com/otcroz/Data_Analysis-Book_checkout_in_Lib/assets/79989242/1eb015c0-05d5-47e7-978a-01bf97e5f9e5)


<details>
<summary>1. 데이터 정제</summary>
<div markdown="1">
  
* 문화 빅데이터 플랫폼에서 제공하는 2023년 공공도서관 대출 정보에서 '서울' 지역에 해당하는 대출내역 데이터 수집, 약 200개의 csv에서 76개의 서울시 대출내역 csv 수집
* 데이터 수집: [공공 도서관 대출정보](https://www.bigdata-culture.kr/bigdata/user/data_market/detail.do?id=d77fa66b-6944-4d8f-b85d-79df6f5ba59e)
* 코드: [data_manipulation.py](https://github.com/otcroz/Data_Analysis-Book_checkout_in_Lib/blob/main/data_manipulation.py)

</div>
</details>

<details>
<summary>2. 데이터 분석</summary>
<div markdown="1">

![슬라이드3](https://github.com/otcroz/Data_Analysis-Book_checkout_in_Lib/assets/79989242/17bd772f-0c0c-457f-81fc-f1188192fa53)

📊데이터 시각화

1) 데이터 분포
![image](https://github.com/otcroz/Data_Analysis-Book_checkout_in_Lib/assets/79989242/58c03723-3d46-4c3f-8380-e367b122a252)

2) 이상치 처리
![image](https://github.com/otcroz/Data_Analysis-Book_checkout_in_Lib/assets/79989242/9e537665-892b-46af-8e76-bbfbd6b0d691)

</div>
</details>

<details>
<summary>3. 예측모델 생성</summary>
<div markdown="1">

![슬라이드4](https://github.com/otcroz/Data_Analysis-Book_checkout_in_Lib/assets/79989242/656dd548-0aa0-46f1-80cb-8b1d5da2a172)
![image](https://github.com/otcroz/Data_Analysis-Book_checkout_in_Lib/assets/79989242/e7ef8746-9481-4765-87bf-e3cf81c6a6e0)

📊데이터 시각화

1) 상관관계 산점도
![상관관계 산점도](https://github.com/otcroz/Data_Analysis-Book_checkout_in_Lib/assets/79989242/ae0ba255-1700-4c94-b920-bccba3882081)

2) 회귀분석 결과 산점도
![회귀분석_linear_산점도](https://github.com/otcroz/Data_Analysis-Book_checkout_in_Lib/assets/79989242/681d1c7c-80b2-4363-bc6a-528fac299522)

3) 랜덤 포레스트 트리 시각화
![random-forest-regression](https://github.com/otcroz/Data_Analysis-Book_checkout_in_Lib/assets/79989242/d3689cf5-354c-4923-866b-75adc9942e57)

4) 최적 파라미터 찾기: 파라미터에 따른 train, test 정확도 측정
![image](https://github.com/otcroz/Data_Analysis-Book_checkout_in_Lib/assets/79989242/421cfc7f-408f-402b-9262-227c78ac8191)


</div>
</details>

![슬라이드6](https://github.com/otcroz/Data_Analysis-Book_checkout_in_Lib/assets/79989242/e0334ff3-183c-4570-889e-d077fc482bcf)
