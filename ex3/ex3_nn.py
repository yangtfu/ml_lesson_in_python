import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.optimize as op
def displayData(x):
  for i in range(100):
    plt.subplot(10,10,i)
    plt.imshow(x[i,:].reshape([20,20]).T,cmap='gray')
    plt.tick_params(labelbottom='off',labelleft='off',bottom='off',left='off',top='off',right='off')
  plt.show()
def sigmoid(z):
  return 1/(1+np.e**(-z))
def predict(theta1,theta2,x):
  a2=sigmoid(x.dot(theta1.T))
  a3=sigmoid(np.concatenate((np.ones([x.shape[0],1]),a2),axis=1).dot(theta2.T))
  p=np.argmax(a3,axis=1)+1
  return p

data=sio.loadmat('ex3data1.mat')
X=data['X']
y=data['y']
thetaData=sio.loadmat('ex3weights.mat')
theta1=thetaData['Theta1']
theta2=thetaData['Theta2']

m=np.size(X,0)
sel=np.random.permutation(X)

displayData(sel)

X=np.concatenate((np.ones([m,1]),X),axis=1)
pred=predict(theta1,theta2,X)

print('Training Set Accuracy:', np.mean(pred==y.flatten()))
