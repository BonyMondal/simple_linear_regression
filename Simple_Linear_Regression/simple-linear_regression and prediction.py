import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
b=pd.read_csv("data1.csv")
x=b.iloc[:, :-1].values
y=b.iloc[:, 1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs experience')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,regressor.predict(x_test),color='blue')
plt.title('salary vs experience')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()
print(regressor.predict([[2.2]]))
print("Intercept=",regressor.intercept_)
print("coefficient=",regressor.coef_)
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))