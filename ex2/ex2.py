## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions 
#  in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
def plotData(X,y):
#  pos=np.where(y==1);neg=np.where(y==0)
  plt.plot(X[y==1,0],X[y==1,1],'k+',linewidth=2,markersize=7,label='Admitted')
  plt.plot(X[y==0,0],X[y==0,1],'ko',markerfacecolor='y',markersize=7,label='Not admitted')
  return None
def sigmoid(z):
  return 1/(1+np.e**(-z))
def costFunction(theta,x,y):
  m=np.size(y)
  theta=np.reshape(theta,[-1,1])
  h=sigmoid(np.dot(x,theta))
  J=1/m*np.sum((-y*np.log(h)-(1-y)*np.log(1-h)))
  grad=1/m*(np.dot(x.T,h-y))
  return J,grad.flatten()
def plotDecisionBoundary(theta,x,y):
  px=np.array([x[:,1].min(),x[:,1].max()])
  py=-1/theta[2]*(theta[1]*px+theta[0])
  plt.plot(px,py,'r-')
  return None
def predict(theta,x):
  g=sigmoid((theta*x).sum(axis=1))
  p=np.zeros(g.shape)
  i=[idx for idx,val in enumerate(g) if val>=0.5]
  p[i]=1
  return p

##Initialization
## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = np.loadtxt('ex2data1.txt',delimiter=',')
X = data[:,0:2]; y = data[:,-1]

## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

print(['Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n'])

plotData(X, y)

# Put some labels 
# Labels and Legend
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

# Specified in plot order
#plt.show()
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in 
#  costFunction.m

#  Setup the data matrix appropriately, and add ones for the intercept term
m, n = np.shape(X)

# Add intercept term to x and X_test
X = np.concatenate((np.ones([m, 1]),X),axis=1)
y = y.reshape([m,1])
# Initialize fitting parameters
initial_theta = np.zeros(n + 1);

# Compute and display initial cost and gradient
cost, grad = costFunction(initial_theta, X, y);

print('Cost at initial theta (zeros):', cost);
print('Gradient at initial theta (zeros):');
print(grad);

#print('\nProgram paused. Press enter to continue.\n');

## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.

#  Run fminunc to obtain the optimal theta
#  This function will return theta and the cost 
result = op.minimize(fun = costFunction, 
                                 x0 = initial_theta, 
                                 args = (X, y),
                                 method='TNC',
                                 jac = True,
                                 options={'maxiter':400},
                                 tol=1e-10
                     )
# Print theta to screen
print('Cost at theta found by fminunc:', result.fun);
print('theta: ');
print(result.x);

# Plot Boundary
plotDecisionBoundary(result.x, X, y);
plt.show()
## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and 
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of 
#  our model.
#
#  Your task is to complete the code in predict.m

#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 

prob = sigmoid((np.array([1, 45, 85])*result.x).sum())
print('For a student with scores 45 and 85, we predict an admission probability of', prob);

# Compute accuracy on our training set
p = predict(result.x, X).reshape(y.shape);
print('Train Accuracy: %f' % ((y[np.where(p == y)].size / float(y.size)) * 100.0))
