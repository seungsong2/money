import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

data = pd.read_csv('./data/1.salary.csv')

# 데이터 전처리 <data "값"만 가져오는 코드>
array = data.values

# X 독립변수 Y 종속 변수로 나누기 위함 / 데이터 분석
# [,] 는 모든 데이터를 뜻함
X = array[:,0]
Y = array[:,1]

# 근속연수 * 연봉
X1 = X.reshape(-1, 1)

# 데이터 분할 //  train Set은 수능 test는 기출문제
(X_train, X_test, Y_train, Y_test) = train_test_split(X1, Y, test_size=0.2)

# 모델 학습
model = LinearRegression()
model.fit(X_train, Y_train)
model.coef_           #연산한 결과를 알려줌
model.fit_intercept   #연산한 결과를 알려줌  방정식을 찾게함

# 예측
y_pred = model.predict(X_test)
print(y_pred)
# 예측 값과 실제 값 오차를 알려줘
error = mean_absolute_error(y_pred, Y_test)
print(error)

# 그래프 그리기 전에 초기화
plt.clf()
# 현실값 산점도
plt.scatter(range(len(Y_test)), Y_test, color='blue', marker='o')
# 예측값
plt.plot(range(len(y_pred)), y_pred, color='r', marker='X')
plt.legend()
plt.xlabel('Experience Years')
plt.ylabel('Salary ($)')
plt.show()