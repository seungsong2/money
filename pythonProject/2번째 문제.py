import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('./data/2.iris.csv')
heder = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width''class' ]


# 데이터 전처리 <data "값"만 가져오는 코드>
array = data.values

# X 독립변수 Y 종속 변수로 나누기 위함 / 데이터 분석
# [,] 는 모든 데이터를 뜻함
X = array[:, 0:4]
Y = array[:, 4]


# 데이터 분할 //  train Set은 수능 test는 기출문제
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.2)

# 모델 학습
model = DecisionTreeClassifier(max_depth=1000, min_samples_split=50, min_samples_leaf=5)
model.fit(X_train, Y_train)
#model.coef_           #연산한 결과를 알려줌
#model.fit_intercept   #연산한 결과를 알려줌  방정식을 찾게함

# 예측
y_pred = model.predict(X_test)
acc = accuracy_score(Y_test, y_pred)
print(acc)



plt.show()
