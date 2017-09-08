#import warmupExercise
#===========Part 1: Baseic function====================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors, ticker, cm
def warmupExercise():
  A=np.eye(5)
  print(A)
def plotData(x,y):
  plt.figure('Figure1')
  plt.plot(x,y,'*')
  plt.xlabel('Popolation of City in 10,000s')
  plt.ylabel('Profit in $10,000s')
def computeCost(x,y,theta):
  J=np.square(np.dot(x,theta)-y).sum()/len(y)/2
  return J
def gradientDescent(x,y,theta,alpha,iterations):
  print('Cost value J:')
  for i in range(iterations):
    theta_j=theta-alpha/len(y)*np.dot(x.T,(np.dot(x,theta)-y))
    theta=theta_j
    print(computeCost(x,y,theta))
  return theta_j

print('Runing warmupExercise...')
print('5x5 Identity Matrix:')

warmupExercise()
print('Program paused. Press Enter to continue.')
input()

#=================== Part 2: Visualizing J(theta0, theta1)========
print('Plotting Data ...')
data = np.loadtxt('ex1data1.txt',delimiter=',')
X = data[:, 0]; y = data[:, 1]
m = y.size #number of training examples

# Plot Data
# Note: You have to complete the code in plotData.m
plotData(X, y)
plt.show()
print('Program paused. Press enter to continue.')
input();
#=================== Part 3: Gradient descent ===================
print('Running Gradient Descent ...\n')

X = np.concatenate((np.ones([m, 1]), data[:,0].reshape(m,1)),axis=1) # Add a column of ones to x
y = y.reshape([-1,1])
theta = np.zeros([2, 1]); # initialize fitting parameters

# Some gradient descent settings
iterations = 1500;
alpha = 0.01;

# compute and display initial cost
computeCost(X, y, theta)

# run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);
# print theta to screen
print('Theta found by gradient descent: ', theta[0], theta[1])

# Plot the linear fit
plotData(X[:,1],y[:,0])
plt.plot(X[:,1],np.dot(X,theta),'-')
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(np.array([[1],[3.5]]).T, theta)[0,0]
print('For population = 35,000, we predict a profit of ', predict1*10000);
predict2 = np.dot(np.array([[1],[7]]).T, theta)[0,0]
print('For population = 70,000, we predict a profit of ', predict2*10000);

print('Program paused. Press enter to continue.\n');
input();

#============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visulaizing J(theta0,theta1)...')
theta0_vals=np.linspace(-10,10,100)
theta1_vals=np.linspace(-1,4,100)

J_vals=np.zeros([theta0_vals.shape[0],theta1_vals.shape[0]])

for i in range(theta0_vals.size):
  for j in range(theta1_vals.size):
    t=np.append(theta0_vals[i],theta1_vals[j]).reshape([2,1])
    J_vals[i,j]=computeCost(X,y,t)
X,Y=np.meshgrid(theta0_vals,theta1_vals)

fig1=plt.figure('surf')
ax=fig1.gca(projection='3d')
surf=ax.plot_surface(X,Y,J_vals.T)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.show()

fig2=plt.figure('contour')
contour=plt.contour(X,Y,J_vals.T)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.plot(theta[0,0],theta[1,0],'rx')
plt.show()
