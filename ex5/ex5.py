import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.optimize as op
def linearRegCostFunction(theta,X,y,lmd):
  m = y.size
  theta=theta.reshape([-1,1])
  J = 1/2/m*np.sum((X.dot(theta)-y)**2)+lmd/m/2*(theta[1:,:]**2).sum()
  grad = 1/m*np.sum((X.dot(theta)-y)*X,axis=0)\
         +np.concatenate((np.zeros([1,1]),lmd/m*theta[1:])).flatten()
  return J,grad

def trainLinearReg(X,y,lmd):
  initial_theta=np.zeros(X.shape[1])
  result=op.minimize(fun=linearRegCostFunction,
                     x0=initial_theta,
                     args=(X,y,lmd),
                     method='TNC',
                     jac=True,
#                     options={'disp':True,'maxiter':200},
#                     tol=1.4e-1
                     )
  return result.x 

def learningCurve(X,y,Xval,yval,lmd):
  m=y.size
  mval=yval.size
  error_train=np.zeros(m)
  error_val=np.zeros(m)
  for i in range(1,m+1):
    theta=trainLinearReg(X[:i,:],y[:i],lmd)
    error_train[i-1],grad=linearRegCostFunction(theta,X[:i,:],y[:i],0)
    error_val[i-1],grad=linearRegCostFunction(theta,Xval,yval,0)
  return error_train,error_val

def polyFeatures(X,p):
  X_poly=np.zeros([X.size,p])
  for i in range(p):
    X_poly[:,i]=(X**(i+1)).flatten()
  return X_poly

def featureNormalize(X):
  mu=np.mean(X,axis=0)
  sigma=np.std(X,axis=0,ddof=1)
  X_norm=(X-mu)/sigma
  return X_norm,mu,sigma

def validationCurve(X,y,Xval,yval):
  lmd_vec=np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
  ml=lmd_vec.size
  error_train=np.zeros(ml)
  error_val=np.zeros(ml)
  for i in range(ml):
    theta=trainLinearReg(X,y,lmd_vec[i])
    error_train[i],grad=linearRegCostFunction(theta,X,y,0)
    error_val[i],grad=linearRegCostFunction(theta,Xval,yval,0)
  return lmd_vec,error_train,error_val

#===========Part 1: Loading and visualizing data=================
print('Loading and visualizing data....')
data=sio.loadmat('ex5data1.mat')
X=data['X']
y=data['y']
Xval=data['Xval']
yval=data['yval']
Xtest=data['Xtest']
ytest=data['ytest']

plt.plot(X,y,'rx',markersize=10,linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')

#===========Part 2: Regularized linear regression cost=========
#===========Part 3: Regularized linear regression gradient======
m=X.shape[0]
theta=np.array([1,1])
J,grad = linearRegCostFunction(theta,np.concatenate((np.ones([m,1]),X),axis=1),y,1)


print('Cost at theta=[1;1] (should be about 303.993192):', J)
print('Grad at theta=[1;1] (should be about [-15.303016; 598.250744]):\n', grad)

#==========Part 4: Traing linear regression===================
lmd=0
theta = trainLinearReg(np.concatenate((np.ones([m,1]),X),axis=1),y,lmd)
plt.plot(X,np.concatenate((np.ones([m,1]),X),axis=1).dot(theta),'--b',linewidth=2)
plt.show()

#==========Part 5: Learning curve for linear regression=======
lmd=0
mval=Xval.shape[0]
error_train,error_val= learningCurve(np.concatenate((np.ones([m,1]),X),axis=1),y,\
                                     np.concatenate((np.ones([mval,1]),Xval),axis=1),yval,lmd)
plt.plot(np.arange(1,m+1),error_train)
plt.plot(np.arange(1,m+1),error_val)
plt.title('Learning curve for linear regression')
plt.legend(("Train","Cross validation"))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0,13,0,150])

print('Training examples \t Train error\t Cross validation')
for i in range(m):
  print('\t%d\t\t%f\t%f' % (i+1,error_train[i],error_val[i]))
plt.show()

#==========Part 6: Feature mapping for polinomial regression==
p=8
mtest=Xtest.shape[0]
X_poly=polyFeatures(X,p)
X_poly,mu,sigma=featureNormalize(X_poly)
X_poly=np.concatenate((np.ones([m,1]),X_poly),axis=1)

X_poly_val=polyFeatures(Xval,p)
X_poly_val=(X_poly_val-mu)/sigma
X_poly_val=np.concatenate((np.ones([mval,1]),X_poly_val),axis=1)

#==========Part 7: Learning curve for polynomial regression==
lmd=0
theta=trainLinearReg(X_poly,y,lmd)

plt.figure(1)
plt.plot(X,y,'rx',markersize=10,linewidth=1.5)
x=np.arange(X.min()-15,X.max()+25,0.05)
X_p=polyFeatures(x,p)
X_p=(X_p-mu)/sigma
X_p=np.concatenate((np.ones([x.shape[0],1]),X_p),axis=1)
plt.plot(x,X_p.dot(theta),'--b',linewidth=2)
plt.xlabel('Change in water level (x)');
plt.ylabel('Water flowing out of the dam (y)');
plt.title(('Polynomial Regression Fit (lambda = %f)' %lmd));

plt.figure(2);
error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lmd);
plt.plot(range(1,m+1), error_train, range(1,m+1), error_val);

plt.title(('Polynomial Regression Learning Curve (lambda = %f)' %lmd));
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 100])
plt.legend(('Train', 'Cross Validation'))

print('Polynomial Regression (lambda = %f)\n' %lmd);
print('Training Examples\tTrain Error\tCross Validation Error');
for i in range(m):
  print('  \t%d\t\t%f\t%f' %(i+1, error_train[i], error_val[i]))
plt.show()

#=========Part 8: Validation for selecting lambda============
lmd_vec,error_train,error_val=validationCurve(X_poly,y,X_poly_val,yval)

plt.figure(3)
plt.plot(lmd_vec,error_train,lmd_vec,error_val)
plt.legend(('Train','Cross validation'))
plt.xlabel('Lambda')
plt.ylabel('Error')

print('lambda\t\tTrain error\tValidation error')

for i in range(lmd_vec.size):
  print(' %f\t%f\t%f'%(lmd_vec[i],error_train[i],error_val[i]))
plt.show()

#========Part 9: Computing test set error 
#and plotting learning curves with randomly selected examples=======
X_poly_test=polyFeatures(Xtest,p)
X_poly_test=(X_poly_test-mu)/sigma
X_poly_test=np.concatenate((np.ones([mtest,1]),X_poly_test),axis=1)

theta=trainLinearReg(X_poly,y,3)
error_val,_=linearRegCostFunction(theta,X_poly_val,yval,0)
error_test,_=linearRegCostFunction(theta,X_poly_test,ytest,0)
print('Test set error is:',error_test)
