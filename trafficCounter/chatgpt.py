"""
pip install -U pandas
pip install -U scikit-learn
"""

import pandas as pd

# 데이터 불러오기
df = pd.read_csv('traffic_data.csv')

# 시간대별 교통량 분석
by_hour = df.groupby('hour')['count'].sum()
print(by_hour)

# 요일별 교통량 분석
by_day = df.groupby('day')['count'].sum()
print(by_day)
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# 특성과 타겟 분리
X = df.drop('count', axis=1)
y = df['count']

# 학습용, 검증용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Decision Tree 모델 학습
tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# 검증용 데이터로 예측 성능 평가
print('검증 세트 점수: {:.2f}'.format(tree.score(X_test, y_test)))