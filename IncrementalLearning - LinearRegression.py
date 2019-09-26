import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


diabetes = datasets.load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(diabetes.data,
                                                    diabetes.target,
                                                    test_size=0.4,
                                                    random_state=0)

def buildMatrix(x):
    [numOfData,feature]=x.shape     
    newX=np.ones((numOfData,feature+1))    
    newX[:,1:]=x
    return newX

def stochasticGD(x,y,epoch,etha):
    [numOfData,feature]=x.shape
    weight=np.zeros((epoch,feature+1)) #+1 for bias
    tempWeight=np.random.randn(1,feature+1)
    tempData=buildMatrix(x)
    for i in range(epoch):        
        for j in range(numOfData):            
            prediction=tempWeight.dot(tempData[j])
            error=prediction - y[j]
            tempWeight=tempWeight-etha*error*tempData[j]           
        weight[i]=tempWeight
    return weight

w=stochasticGD(x_train,y_train,10,0.02)
print(w)

