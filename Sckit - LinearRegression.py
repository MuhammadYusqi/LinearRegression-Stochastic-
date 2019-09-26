import numpy as np
import matplotlib.pyplot as plot
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


diabetes = datasets.load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(diabetes.data,
                                                    diabetes.target,
                                                    test_size=0.4,
                                                    random_state=0)
model = LinearRegression()
model.fit(x_train, y_train)

coef = model.coef_
intercept = model.intercept_
prediction = model.predict(x_test)
accuracy=r2_score(y_test,prediction)

print("Coef = ",coef)
print("Intercept = ",intercept)
print("Accuracy Score = ", accuracy)

plot.title('Housing Price Prediction')
plot.xlabel('Area')
plot.ylabel('Price in Million($)')
plot.plot(x_train, y_train, 'k.')
plot.plot(y_test,prediction,'r.')
plot.grid(True)
plot.show()


