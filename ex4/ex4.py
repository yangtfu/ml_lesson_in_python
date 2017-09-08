import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.optimize as op
def displayData(x):
  m=x.shape[0]
  n=np.sqrt(m)
  for i in range(m):
    plt.subplot(n,n,i)
    plt.imshow(x[i,:].reshape([20,20]).T,cmap='gray')
    plt.tick_params(labelbottom='off',labelleft='off',bottom='off',left='off',top='off',right='off')
  plt.show()

def sigmoid(z):
  return 1/(1+np.e**(-z))

def sigmoidGradient(z):
  return sigmoid(z)*(1-sigmoid(z))

def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lmd):
  #=====cost
  m,n=X.shape
  i,j,k=input_layer_size,hidden_layer_size,num_labels
  theta1=nn_params[:(i+1)*j].reshape([j,i+1])
  theta2=nn_params[(i+1)*j:].reshape([k,j+1])
  h1=np.concatenate((np.ones([m,1]),sigmoid(X.dot(theta1.T))),axis=1)
  h=sigmoid(h1.dot(theta2.T))
  ym=np.zeros([m,num_labels])
  for l in range(1,num_labels+1):
    ym[:,l-1]=((y==l)+0.0).flatten()
  J=-1/m*(ym*np.log(h)+(1-ym)*np.log(1-h)).sum()\
    +lmd/2/m*((theta1[:,1:]**2).sum()+(theta2[:,1:]**2).sum())
  #======grad
  Delta2=np.zeros([j+1,k])
  Delta1=np.zeros([i+1,j])
  for t in range(m):
    a1=X[t,:].reshape([1,i+1])
    z2=a1.dot(theta1.T)
    a2=np.concatenate((np.ones([1,1]),sigmoid(z2)),axis=1)
    z3=a2.dot(theta2.T)
    a3=sigmoid(z3)

    delta3=a3-ym[t,:].reshape([1,k])
    delta2=(delta3.dot(theta2))[:,1:]*sigmoidGradient(z2)
    
    Delta2=Delta2+(a2.T).dot(delta3)
    Delta1=Delta1+(a1.T).dot(delta2)
  D2=(Delta2/m+lmd/m*np.concatenate((np.zeros([1,Delta2.shape[1]]),theta2.T[1:,:]),axis=0)).T
  D1=(Delta1/m+lmd/m*np.concatenate((np.zeros([1,Delta1.shape[1]]),theta1.T[1:,:]),axis=0)).T
  grad=np.concatenate((D1.flatten(),D2.flatten()))
  return J,grad

def computeNumericalGradient(J, theta):
  numgrad=np.zeros(theta.shape)
  perturb=np.zeros(theta.shape)
  e=1e-4
  for p in range(theta.size):
    perturb[p]=e
    loss1=J(theta-perturb)[0]
    loss2=J(theta+perturb)[0]
    numgrad[p]=(loss2-loss1)/2/e
    perturb[p]=0
  return numgrad

def randInitWeights(i,j):
  epsilon_init=0.12
  return np.random.random([j,i+1])*2*epsilon_init-epsilon_init

def checkNNGradients(lmd=0):
  ils,hls,nl=3,5,3
  m=5
  def debugInit(fan_out,fan_in):
    W=np.zeros([fan_out,1+fan_in])
    W=np.reshape(np.sin(np.arange(1,1+W.size)),W.T.shape).T/10
    return W
  theta1=debugInit(hls,ils)
  theta2=debugInit(nl,hls)
  X=np.concatenate((np.ones([m,1]),debugInit(m,ils-1)),axis=1)
  y=(1+np.mod(np.arange(1,m+1),nl))
  nn_params=np.concatenate((theta1.flatten(order='C'),theta2.flatten(order='C')))
  def costFunc(p,ils1=ils,hls1=hls,nl1=nl,X1=X,y1=y,lmd1=lmd):
    return nnCostFunction(p,ils1,hls1,nl1,X1,y1,lmd1)
  cost,grad=costFunc(nn_params)
  numgrad=computeNumericalGradient(costFunc,nn_params)
  print('numgrad = \n',numgrad)
  print('grad = \n', grad)
  diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
  print('The relative difference is:', diff)

def predict(theta1,theta2,x):
  a2=sigmoid(x.dot(theta1.T))
  a3=sigmoid(np.concatenate((np.ones([x.shape[0],1]),a2),axis=1).dot(theta2.T))
  p=np.argmax(a3,axis=1)+1
  return p 
  
input_layer_size=400
hidden_layer_size=25
num_labels=10
#==========Load and visualize data=======
data=sio.loadmat('ex4data1.mat')
X=data['X']
y=data['y']

m=np.size(X,0)
sel=np.random.permutation(X)
#displayData(sel)

X=np.concatenate((np.ones([m,1]),X),axis=1)
#y[y==10]=0
#=========Load parameters===================
thetaData=sio.loadmat('ex4weights.mat')
theta1=thetaData['Theta1'].flatten()
theta2=thetaData['Theta2'].flatten()
nn_params=np.concatenate((theta1[:],theta2[:]))

#========Compute cost======================
lmd=0
J,grad=nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lmd)
print('Cost at parameters(loaded from ex4weights):',J)
#========Implement regularization=========

lmd=1
J,grad=nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lmd)
print('Cost at parameters with lambda=1 (loaded from ex4weights):',J)

#========Sigmoid gradient================
print('Evaluating sigmoid gradient')

g=sigmoidGradient(np.array([1,-0.5,0,0.5,1]))
print('Sigmoid gradient evaluated at [1,-0.5,0,0.5,1]:')
print(g)

#========Initialize parameters==========
print('Initializing Neural Network parameters...')
initial_theta1=randInitWeights(input_layer_size,hidden_layer_size)
initial_theta2=randInitWeights(hidden_layer_size,num_labels)
initial_nn_params=np.concatenate((initial_theta1.flatten(),initial_theta2.flatten()))

#========Implement Backpropagation======
print('Checking Backpropagation')
lmd=3
checkNNGradients(lmd)

#=======Implement Regularizztion=======
debug_J,debug_grad=nnCostFunction(nn_params,input_layer_size,\
                                  hidden_layer_size,num_labels,X,y,lmd)

print('Cost at debugging parameter(should be about 0.576051):',debug_J)

#=======Training NN===================
print('Training NN....')
lmd=1
result = op.minimize(fun=nnCostFunction,
                     x0=initial_nn_params,
                     args=(input_layer_size,hidden_layer_size,num_labels,X,y,lmd),
                     method='TNC',
                     jac=True,
                     options={'disp':True,'maxiter':300},
                     tol=1e-5)
theta1=result.x[:(hidden_layer_size*(input_layer_size+1))]\
               .reshape([hidden_layer_size,input_layer_size+1])
theta2=result.x[(hidden_layer_size*(input_layer_size+1)):]\
               .reshape([num_labels,hidden_layer_size+1])
displayData(theta1[:,1:])
#========predict=====================
pred=predict(theta1,theta2,X)
print('Training set accuracy is:', np.mean(pred==y.flatten())*100)



