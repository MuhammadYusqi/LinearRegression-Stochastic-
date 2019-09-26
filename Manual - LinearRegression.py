from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def ratarata(lis):
    hasil=0
    for i in range(len(lis)):
        hasil+=lis[i]
    hasil=hasil/len(lis)
    return hasil

def pangkat(a,b):
    hasil = a**b
    return hasil

def var(listX):
    tmp=0
    for i in range(len(listX)):
        tmp+=pangkat((listX[i]-ratarata(listX)),2)
    hasil = tmp / (len(listX)-1)
    return hasil

def cov(listX,listY):
    tmp=0
    for i in range(listX):
        tmp+=(listX[i]-ratarata(listX))*(listY-ratarata(listY))
    hasil = tmp / (len(listX)-1)
    return hasil

def w1(listX,listY):
    hasil = cov(listX,listY)/ var(listX)
    return hasil

def w0(listX,listY):
    hasil = ratarata(listY) - (w1(listX,listY)*ratarata(listX))
    return hasil

diabetes = datasets.load_boston()
listX, listY, y_train, y_test = train_test_split(diabetes.data,
                                                    diabetes.target,
                                                    test_size=0.4,
                                                    random_state=0)
print(cov(listX,listY))
