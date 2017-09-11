import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.optimize as op
def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lmd):
  X=params[:num_movies*num_features].reshape([num_movies,num_features])
  theta=params[num_movies*num_features:].reshape([num_users,num_features])
  J=np.sum(0.5*((X.dot(theta.T)-Y)*R)**2)+lmd/2*(np.sum(theta**2)+np.sum(X**2))
  grad_x=((X.dot(theta.T)-Y)*R).dot(theta)+lmd*X
  grad_theta=((X.dot(theta.T)-Y)*R).T.dot(X)+lmd*theta
  grad=np.hstack((grad_x.flatten(),grad_theta.flatten()))
  return J,grad
def checkCostFunction(lmd=0):
  X_t=np.random.rand(4,3)
  Theta_t=np.random.rand(5,3)

  Y=X_t.dot(Theta_t.T)
  Y[np.random.rand(Y.shape[0],Y.shape[1])>0.5]=0
  R=np.zeros(Y.shape)
  R[Y!=0]=1

  X=np.random.randn(X_t.shape[0],X_t.shape[1])
  Theta=np.random.randn(Theta_t.shape[0],Theta_t.shape[1])
  num_users=Y.shape[1]
  num_movies=Y.shape[0]
  num_features=Theta_t.shape[1]
  func=lambda t:cofiCostFunc(t,Y,R,num_users,num_movies,num_features,lmd)
  def computeNumericalGradient(J,theta):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e=1e-4
    for p in range(theta.size):
      perturb[p]=e
      loss1,_=J(theta-perturb)
      loss2,_=J(theta+perturb)
      numgrad[p]=(loss2-loss1)/2/e
      perturb[p]=0
    return numgrad
  numgrad = computeNumericalGradient(func,np.hstack((X.flatten(),Theta.flatten())))
  cost,grad=cofiCostFunc(np.hstack((X.flatten(),Theta.flatten())),Y,R,num_users,num_movies,num_features,lmd)
  for i in range(numgrad.size):
    print('\t%f\t%f' %(numgrad[i],grad[i]))
  print('The above two columns you get should be very similar.\n' \
         '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

  diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
  print('If your backpropagation implementation is correct, then \n' \
         'the relative difference will be small (less than 1e-9). \n' \
         'Relative Difference: %g'  %diff);
## Machine Learning Online Class
#  Exercise 8 | Anomaly Detection and Collaborative Filtering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     estimateGaussian.m
#     selectThreshold.m
#     cofiCostFunc.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## =============== Part 1: Loading movie ratings dataset ================
#  You will start by loading the movie ratings dataset to understand the
#  structure of the data.
#  
print('Loading movie ratings dataset.\n')

#  Load data
data = sio.loadmat ('ex8_movies.mat')
Y,R=data['Y'],data['R']
#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
#  943 users
#
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i

#  From the matrix, we can compute statistics like average rating.
print('Average rating for movie 1 (Toy Story): %f / 5\n' %np.mean(Y[0, R[0, :]==1]))

#  We can "visualize" the ratings matrix by plotting it with imagesc
plt.imshow(Y)
plt.ylabel('Movies')
plt.xlabel('Users')

## ============ Part 2: Collaborative Filtering Cost Function ===========
#  You will now implement the cost function for collaborative filtering.
#  To help you debug your cost function, we have included set of weights
#  that we trained on that. Specifically, you should complete the code in 
#  cofiCostFunc.m to return J.

#  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
data = sio.loadmat ('ex8_movieParams.mat')
X,Theta,num_users,num_movies,num_features=\
data['X'],data['Theta'],data['num_users'],data['num_movies'],data['num_features']
#  Reduce the data set size so that this runs faster
num_users = 4; num_movies = 5; num_features = 3
X = X[:num_movies, :num_features]
Theta = Theta[:num_users, :num_features]
Y = Y[:num_movies, :num_users]
R = R[:num_movies, :num_users]

#  Evaluate cost function
J,_ = cofiCostFunc(np.hstack((X.flatten(),Theta.flatten())), Y, R, num_users, num_movies,\
               num_features, 0)
           
print('Cost at loaded parameters: %f '\
         '\n(this value should be about 22.22)' %J)

## ============== Part 3: Collaborative Filtering Gradient ==============
#  Once your cost function matches up with ours, you should now implement 
#  the collaborative filtering gradient function. Specifically, you should 
#  complete the code in cofiCostFunc.m to return the grad argument.
#  
print('Checking Gradients (without regularization) ... ')

#  Check gradients by running checkNNGradients
checkCostFunction()

## ========= Part 4: Collaborative Filtering Cost Regularization ========
#  Now, you should implement regularization for the cost function for 
#  collaborative filtering. You can implement it by adding the cost of
#  regularization to the original cost computation.
#  

#  Evaluate cost function
J,_ = cofiCostFunc(np.hstack((X.flatten(),Theta.flatten())), Y, R, num_users, num_movies,num_features, 1.5)
           
print('Cost at loaded parameters (lambda = 1.5): %f '\
         '(this value should be about 31.34)\n' %J);

## ======= Part 5: Collaborative Filtering Gradient Regularization ======
#  Once your cost matches up with ours, you should proceed to implement 
#  regularization for the gradient. 
#

#  
print('Checking Gradients (with regularization) ... ');

#  Check gradients by running checkNNGradients
checkCostFunction(1.5)

## ============== Part 6: Entering ratings for a new user ===============
#  Before we will train the collaborative filtering model, we will first
#  add ratings that correspond to a new user that we just observed. This
#  part of the code will also allow you to put in your own ratings for the
#  movies in our dataset!
#
movieList = dict()
with open('movie_ids.txt',encoding = "ISO-8859-1") as f:
  for line in f.readlines():
    l=line.split(maxsplit=1)
    movieList[int(l[0])-1]=l[1]

#  Initialize my ratings
my_ratings = np.zeros([1682, 1])

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[0] = 4;

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[97] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[6] = 3
my_ratings[11]= 5
my_ratings[53] = 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354]= 5

print('\nNew user ratings:')
for i in range(my_ratings.size):
  if my_ratings[i] > 0: 
    print('Rated %d for %s'% (my_ratings[i], \
                 movieList[i]),end='')

## ================== Part 7: Learning Movie Ratings ====================
#  Now, you will train the collaborative filtering model on a movie rating 
#  dataset of 1682 movies and 943 users
#

print('Training collaborative filtering...');

#  Load data
data=sio.loadmat('ex8_movies.mat')
Y=data['Y']
R=data['R']
#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
#  943 users
#
#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i

#  Add our own ratings to the data matrix
Y = np.hstack((my_ratings,Y))
R = np.hstack(((my_ratings != 0),R))

#  Normalize Ratings
Ymean =np.mean(Y*R,axis=1)

#  Useful Values
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10

# Set Initial Parameters (Theta, X)
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

initial_parameters = np.hstack((X.flatten(), Theta.flatten()))

# Set Regularization
lmd = 10;
result=op.minimize(fun=cofiCostFunc,
                   x0=initial_parameters,
                   args=(Y,R,num_users,num_movies,num_features,lmd),
                   method='TNC',
                   jac=True,
                   options={'disp':True,'maxiter':100},
#                   tol=1.4e-1
                   )
# Unfold the returned theta back into U and W
theta=result.x
X = theta[:num_movies*num_features].reshape([num_movies, num_features])
Theta = theta[num_movies*num_features:].reshape([num_users, num_features])

print('Recommender system learning completed.\n');

## ================== Part 8: Recommendation for you ====================
#  After training the model, you can now make recommendations by computing
#  the predictions matrix.
#

p = X.dot(Theta.T)
my_predictions = p[:,0] + Ymean;

r, ix = -np.sort(-my_predictions),np.argsort(-my_predictions)
print('Top recommendations for you:');
for i in range(10):
  j = ix[i]
  print('Predicting rating %.1f for movie %s' %(my_predictions[j],\
            movieList[j]),end='')

print('\nOriginal ratings provided:');
for i in range(my_ratings.size):
  if my_ratings[i] > 0:
    print('Rated %d for %s' %(my_ratings[i], movieList[i]),end='')
