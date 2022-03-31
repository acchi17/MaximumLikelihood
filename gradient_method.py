from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np


def logLikelihoodFunc(x, mu, sig):
    p = 1/np.sqrt(np.pi * sig **2)*np.exp(-(x-mu)**2 /(2*sig**2))
    return np.sum(np.log(p))

# f: function to differentiate
# x: derivative variable
def calc_gradient(f, x):
    h=1e-4
    x_1 = np.copy(x)
    x_1 = x_1 + h
    diff = (f(x_1) - f(x))/h
    return diff

# f:   likelihood function
# x:   parameter to learn
# eta: learning rate
def Learning(f, x, eta):
    grads = calc_gradient(f,x)
    params=np.array([])
    likelyfood=np.array([])
    for i in range(10000):
        x_1=np.copy(x)
        LF_old=f(x)
        grads=gradient_scalar(f,x_1)
        x=x_1+eta*grads
        params=np.append(params,x)
        LF=f(x)
        likelyfood=np.append(LF,likelyfood)
    return params, likelyfood


def main():
    iris = load_iris()
    #print(iris.data.shape)   # (150, 4) 150: number of iris 4features data , 4: number of iris features   
    #print(iris.target.shape) # (150,) 150: number of iris class data, 0:setosa, 1; versicolor, 2:virginica

    data = iris.data[iris.target==1]
    plt.hist(data[:, 1], bins=10, ec='black')
    plt.show()
