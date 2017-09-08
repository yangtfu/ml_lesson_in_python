#import warmupExercise
#===========Part 1: Baseic function====================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors, ticker, cm
def featureNormalize(X):
  mu=np.mean(X,axis=0)
  sigma=np.std(X,axis=0)
  X=(X-np.ones(np.shape(X))*mu)/sigma
  return X,mu,sigma
def computeCost(x,y,theta):
  J=np.square(np.dot(x,theta)-y).sum()/len(y)/2
  return J
def gradientDescent(x,y,theta,alpha,iterations):
  J_history=np.zeros(iterations)
  for i in range(iterations):
    theta_j=theta-alpha/len(y)*np.dot(x.T,(np.dot(x,theta)-y))
    theta=theta_j
    J_history[i]=computeCost(x,y,theta)
  return theta_j,J_history
def normalEqn(X,y):
  theta=np.dot(np.dot(np.linalg.pinv(np.dot(X.T,X)),X.T),y)
  return theta

print('Loading Data ...')
data = np.loadtxt('ex1data2.txt',delimiter=',')
X = data[:, 0:2]; y = data[:, 2]
m = y.size #number of training examples

# Plot Data
# Note: You have to complete the code in plotData.m
print('First 10 examples from the dataset:')
print(' x=\n ',X[1:10,0:2])
print(' y=\n',y[0:10].reshape([-1,1]))

print('Normalizing Features...')
X, mu, sigma = featureNormalize(X)
X = np.concatenate((np.ones([m, 1]), X),axis=1) # Add a column of ones to x
y = y.reshape([-1,1])
theta = np.zeros([3, 1]); # initialize fitting parameters

# Some gradient descent settings
iterations = 8500;
alpha = 0.01;

# compute and display initial cost
computeCost(X, y, theta)

# run gradient descent
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
plt.plot(range(iterations),J_history,'-')
plt.show()
# print theta to screen
print('Theta found by gradient descent: \n', theta)
price =np.dot(np.append(np.array([1,]), (np.array([1650,3])-mu)/sigma),theta)
print('Price computed from gradient descent: ', price)

data = np.loadtxt('ex1data2.txt',delimiter=',')
X = data[:, 0:2]; y = data[:, 2]
X = np.concatenate((np.ones([m, 1]), X),axis=1) # Add a column of ones to x
y = y.reshape([-1,1])
theta=normalEqn(X, y)
print('Theta found by gradient descent: \n', theta)
price =np.dot(np.append(np.array([1,]), np.array([1650,3])),theta)
print('Price computed from the normal equations:',price)
