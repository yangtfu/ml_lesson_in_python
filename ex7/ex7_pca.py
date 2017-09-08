import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import matplotlib.pyplot as plt
import random
def featureNormalize(X):
  mu=np.mean(X,axis=0).reshape([-1,X.shape[1]])
  sigma=np.std(X,axis=0,ddof=1).reshape([-1,X.shape[1]])
  X_norm=(X-mu)/sigma
  return X_norm,mu.flatten(),sigma.flatten()
def projectData(X_norm,U,K):
  Z=X_norm.dot(U[:K,:].T)
  return Z
def recoverData(Z,U,K):
  X=Z.dot(U[:K,:])
  return X
def displayData(x):
  plt.figure()
  r=int(x.shape[0]**0.5//1)
  px=int(x.shape[1]**0.5)
  py=int(x.shape[1]//px)
  for i in range(r**2):
    plt.subplot(r,r,i)
    plt.imshow(x[i,:].reshape([px,py]).T,cmap='gray')
    plt.tick_params(labelbottom='off',labelleft='off',bottom='off',left='off',top='off',right='off')
  plt.show()
def findClosestCentroids(X,initial_centroids):
  dist=np.zeros([X.shape[0],initial_centroids.shape[0]])
  for i in range(initial_centroids.shape[0]):
    dist[:,i]=((X-initial_centroids[i,:])**2).sum(axis=1)
  idx=np.argmax(-dist,axis=1)
  return idx

def computeCentroids(X,idx,K):
  centroids=np.zeros([K,X.shape[1]])
  for i in range(K):
    centroids[i,:] = (X[idx==i].sum(axis=0))/np.count_nonzero(idx==i)
  return centroids

def runkMeans(X,initial_centroids,max_iters,draw):
  if draw:
    plt.interactive(True)
    plt.figure()
  for i in range(max_iters):
    idx=findClosestCentroids(X,initial_centroids)
    centroids_old=initial_centroids
    initial_centroids=computeCentroids(X,idx,K)
    if draw:
      dots=plt.scatter(X[:,0],X[:,1])
      color=['r' if i==0 else 'b' if i==1 else 'g' for i in idx]
      dots.set_color(color)
      plt.plot(initial_centroids[:,0],initial_centroids[:,1],'kx',markersize=8,linewidth=2)
    for j in range(initial_centroids.shape[0]):
      x=np.array([centroids_old[j,0],initial_centroids[j,0]])
      y=np.array([centroids_old[j,1],initial_centroids[j,1]])
      if draw:
        plt.plot(x,y,'k-')
    if ((centroids_old==initial_centroids).all()): break
    if draw: plt.pause(0.5)
  return initial_centroids,idx

def kMeansInitCentroids(X,K):
  idx=np.random.permutation(X)
  centroids=idx[:K]
  return centroids
## Machine Learning Online Class
#  Exercise 7 | Principle Component Analysis and K-Means Clustering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     pca.m
#     projectData.m
#     recoverData.m
#     computeCentroids.m
#     findClosestCentroids.m
#     kMeansInitCentroids.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easily to
#  visualize
#
print('Visualizing example dataset for PCA.')

#  The following command loads the dataset. You should now have the 
#  variable X in your environment
data=sio.loadmat('ex7data1.mat')
X=data['X']

#  Visualize the example dataset
plt.interactive(True)
plt.plot(X[:, 0], X[:, 1], 'bo')
plt.axis([0.5,6.5,2,8])
#axis square;

## =============== Part 2: Principal Component Analysis ===============
#  You should now implement PCA, a dimension reduction technique. You
#  should complete the code in pca.m
#
print('\nRunning PCA on example dataset.\n')

#  Before running PCA, it is important to first normalize X
X_norm, mu, sigma = featureNormalize(X)

#  Run PCA
u,s,v = np.linalg.svd(X_norm);

#  Compute mu, the mean of the each feature

#  Draw the eigenvectors centered at mean of data. These lines show the
#  directions of maximum variations in the dataset.
x1=np.array([mu[0],mu[0]+1.5*s[0]**2/50*v[0,0]])
y1=np.array([mu[1],mu[1]+1.5*s[0]**2/50*v[0,1]])
x2=np.array([mu[0],mu[0]+1.5*s[1]**2/50*v[1,0]])
y2=np.array([mu[1],mu[1]+1.5*s[1]**2/50*v[1,1]])
plt.plot(x1,y1, '-k', linewidth=2);
plt.plot(x2,y2, '-k', linewidth=2);
plt.axis('equal')

print('Top eigenvector: ');
print(' V(:,0) = ', v[0,:]);
print('(you should expect to see -0.707107 -0.707107)');


## =================== Part 3: Dimension Reduction ===================
#  You should now implement the projection step to map the data onto the 
#  first k eigenvectors. The code will then plot the data in this reduced 
#  dimensional space.  This will show you what the data looks like when 
#  using only the corresponding eigenvectors to reconstruct it.
#
#  You should complete the code in projectData.m
#
print('\nDimension reduction on example dataset.')

#  Plot the normalized dataset (returned from pca)
plt.figure()
plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo')
plt.axis([-4,3,-4,3])
plt.axis('equal')

#  Project the data onto K = 1 dimension
K = 1
Z = projectData(X_norm, v, K)
print('Projection of the first example:', Z[0])
print('(this value should be about 1.481274)\n');

X_rec  = recoverData(Z, v, K)
print('Approximation of the first example: %f %f' %(X_rec[0,0],X_rec[0,1]))
print('(this value should be about  -1.047419 -1.047419)\n')

#  Draw lines connecting the projected points to the original points
plt.plot(X_rec[:,0] , X_rec[:,1], 'ro')

## =============== Part 4: Loading and Visualizing Face Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment
#
print('Loading face dataset.\n');

#  Load Face dataset
data2=sio.loadmat('ex7faces.mat')
X=data2['X']
#  Display the first 100 faces in the dataset
displayData(X[:100, :]);

## =========== Part 5: PCA on Face Data: Eigenfaces  ===================
#  Run PCA and visualize the eigenvectors which are in this case eigenfaces
#  We display the first 36 eigenfaces.
#
print('Running PCA on face dataset.' '(this mght take a minute or two ...)\n');

#  Before running PCA, it is important to first normalize X by subtracting 
#  the mean value from each feature
X_norm, mu, sigma = featureNormalize(X)

#  Run PCA
U,S,V = np.linalg.svd(X_norm)

#  Visualize the top 36 eigenvectors found
displayData(V[:36,:])

## ============= Part 6: Dimension Reduction for Faces =================
#  Project images to the eigen space using the top k eigenvectors 
#  If you are applying a machine learning algorithm 
print('Dimension reduction for face dataset.\n');

K = 100;
Z = projectData(X_norm, V, K);

print('The projected data Z has a size of: ',Z.shape[1])

## ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
#  Project images to the eigen space using the top K eigen vectors and 
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed

print('Visualizing the projected (reduced dimension) faces.\n');

K = 100;
X_rec  = recoverData(Z, V, K);

# Display reconstructed data from only k eigenfaces
displayData(X_rec[:100,:])

## === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
#  One useful application of PCA is to use it to visualize high-dimensional
#  data. In the last K-Means exercise you ran K-Means on 3-dimensional 
#  pixel colors of an image. We first visualize this output in 3D, and then
#  apply PCA to obtain a visualization in 2D.


# Re-load the image from the previous exercise and run K-Means on it
# For this to work, you need to complete the K-Means assignment first

#  Load an image of a bird
A = plt.imread('bird_small.png')
img_size = A.shape
X = A.reshape([-1, 3])
K = 16 
max_iters = 10
initial_centroids = kMeansInitCentroids(X, K)
centroids, idx = runkMeans(X, initial_centroids, max_iters,False)

#  Sample 1000 random indexes (since working with all the data is
#  too expensive. If you have a fast computer, you may increase this.
sel = random.sample(range(X.shape[0]),1000) 

#  Visualize the data and centroid memberships in 3D
color=np.zeros(1000)
for i,k in enumerate(idx[sel]):
  color[i]=k*10
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2],c=color,cmap=plt.cm.hsv)

## === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
# Use PCA to project this cloud to 2D for visualization

# Subtract the mean to use PCA
X_norm, mu, sigma = featureNormalize(X)

# PCA and project the data to 2D
U,S,V = np.linalg.svd(X_norm)
Z = projectData(X_norm, V, 2)

# Plot in 2D
plt.figure()
dots=plt.scatter(Z[sel,0],Z[sel,1],c=color,cmap=plt.cm.hsv)
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
