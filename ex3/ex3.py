import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.optimize as op
def displayData(x):
  r=int(x.shape[0]**0.5//1)
  for i in range(r**2):
    plt.subplot(r,r,i)
    plt.imshow(x[i,:].reshape([20,20]).T,cmap='gray')
    plt.tick_params(labelbottom='off',labelleft='off',bottom='off',left='off',top='off',right='off')
  plt.show()

def sigmoid(z):
  return 1/(1+np.e**(-z))
def lrCostFunction(theta,x,y,ld):
  m=np.size(y)
  theta=np.reshape(theta,[-1,1])
  y=np.reshape(y,[-1,1])
  h=sigmoid(np.dot(x,theta))
  J=1/m*np.sum((-y*np.log(h)-(1-y)*np.log(1-h)))+ld/2/m*np.sum(theta[1:]**2)
  grad=1/m*(np.dot(x.T,h-y))+np.concatenate((np.zeros([1,1]),ld/m*theta[1:]))
  return J,grad.flatten()
def oneVsAll(X,y,num,lmd):
  m,n=X.shape
  initial_theta=np.zeros(n)
  theta=np.zeros([n,num])
  for i in range(num):    
    print('Training number ',i)
    yi=(y==i)+0.0
    result=op.minimize(fun=lrCostFunction,
                     x0=initial_theta,
                     args=(X,yi,lmd),
                      method='TNC',
                     jac=True,
#                     options={'disp':True,'maxiter':50},
                     tol=1e-10)
    theta[:,i]=result.x
  return theta 
def predictOneVsAll(theta,x):
  g=sigmoid(x.dot(theta))
  p=np.argmax(g,axis=1)
  return p

data=sio.loadmat('ex3data1.mat')
X=data['X']
y=data['y']
y[y==10]=0

m=np.size(X,0)
sel=np.random.permutation(X)

displayData(sel[:100,:])

print('Training One-vs-All Logistic Regression...')

lmd=0.1
X=np.concatenate((np.ones([m,1]),X),axis=1)
theta=oneVsAll(X,y,10,lmd)

pred=predictOneVsAll(theta,X)

print('Training Set Accuracy:', np.mean(pred==y.flatten()))
