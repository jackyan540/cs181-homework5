# Starter code for use with autograder.
import numpy as np
import matplotlib.pyplot as plt


def get_cumul_var(mnist_pics,
                  num_leading_components=500):

    """
    Perform PCA on mnist_pics and return cumulative fraction of variance
    explained by the leading k components.

    Returns:
        A (num_leading_components, ) numpy array where the ith component
        contains the cumulative fraction (between 0 and 1) of variance explained
        by the leading i components.

    Args:

        mnist_pics, (N x D) numpy array:
            Array containing MNIST images.  To pass the test case written in
            T5_P2_Autograder.py, mnist_pics must be a 2D (N x D) numpy array,
            where N is the number of examples, and D is the dimensionality of
            each example.

        num_leading_components, int:
            The variable representing k, the number of PCA components to use.
    """

    # TODO: compute PCA on input mnist_pics
    mean_x = np.mean(mnist_pics, axis=0)
    recentered_data = mnist_pics - mean_x
    covariance = np.cov(mnist_pics.T)
    eig_vals, eig_vecs = np.linalg.eig(covariance)
    eig_vals = np.real(eig_vals)
    eig_vecs = np.real(eig_vecs)
    sort_indices = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[sort_indices]
    eig_vecs = eig_vecs[:,sort_indices]

    # TODO: return a (num_leading_components, ) numpy array with the cumulative
    # fraction of variance for the leading k components
    ret = np.zeros(num_leading_components)
    tot_variance = np.sum(eig_vals)
    for i in range(num_leading_components):
        ret[i] = np.sum(eig_vals[0:i+1])/tot_variance
        
    # Plot Eigenvalues for most significant 500 components
    plt.figure()
    plt.plot(np.arange(500), eig_vals[0:500])
    plt.title("Eigenvalues for Most Significant 500 Components")
    plt.xlabel("Components Index (from most significant to least)")
    plt.ylabel("Eigenvalue")
    plt.savefig("2_1_1.png")
    
    # Plot Cumulative Proportion of Variance
    plt.figure()
    plt.plot(np.arange(500),ret)
    plt.title("Cumulative Proportion of Variance explained by the first k most significant components")
    plt.xlabel("k")
    plt.ylabel("Cumulative Proportion of Variance")
    plt.savefig("2_1_2.png")
    
    # Plot Mean Image of the Dataset
    plt.figure()
    plt.imshow(mean_x.reshape(28,28), cmap='Greys_r')
    plt.savefig("2_2_1.png")
    
    # Plot Image Corresponding to each of the first 10 principal components
    plt.figure()
    fig, ax = plt.subplots(5,2, figsize=(10, 30))
    axs = ax.ravel()
    for i, a in enumerate(axs):
        a.imshow(eig_vecs[:,i].reshape(28,28), cmap='Greys_r')
        a.set_title("Principal Component " + str(i+1))
    plt.savefig("2_2_2.png")
    
    # Compute Reconstruction Error using the mean image
    total = 0
    for x in mnist_pics:
        total += np.linalg.norm(x - mean_x)**2
    total = total/(mnist_pics.shape[0])
    print("The Reconstruction Error when using the mean image is: " + str(total))
    
    # Compute Reconstruction error using the first 10 principal components
    total = 0
    for i in range(mean_x.shape[0]-10):
        total += eig_vecs[:,i+10].T @ covariance @ eig_vecs[:,i+10]
    print("The Reconstruction Error when using the first 10 principal components is: " + str(total))
    
    return ret

# Load MNIST.
mnist_pics = np.load("data/images.npy")

# Reshape mnist_pics to be a 2D numpy array.
num_images, height, width = mnist_pics.shape
mnist_pics = np.reshape(mnist_pics, newshape=(num_images, height * width))

num_leading_components = 500

cum_var = get_cumul_var(
    mnist_pics=mnist_pics,
    num_leading_components=num_leading_components)

# Example of how to plot an image.
#plt.figure()
#plt.imshow(mnist_pics[0].reshape(28,28), cmap='Greys_r')
#plt.show()

# CS 181, Spring 2020
# Homework 4: K-means

import numpy as np
import matplotlib.pyplot as plt
from seaborn import heatmap

class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K):
        self.K = K

    def objective(self, full=False):
        elements = self.distances[np.arange(self.X.shape[0]), self.assignments]**2
        return np.sum(elements)

    def __update_means(self):
        # For each cluster k...
        for k in range(self.K):
            # Get points assigned to k
            in_class_values = self.X[self.assignments == k]
            if sum(self.assignments == k) == 0: continue # mean of empty class is undefined
            # Update mean of cluster k to be mean of points assigned to it
            self.means[k] = np.mean(in_class_values, axis=0)

    def __initialize_means(self):
        self.means = np.random.randn(self.K, self.X.shape[1]) # K x 784

    # Returns an N x K array, where Distances_nk is the distance from point n to mean of cluster k
    def __getdistances(self):
        distances = np.zeros((self.X.shape[0], self.K)) # N x K
        # For each cluster k...
        for k in range(self.K):
            # Find Euclidean distance between cluster k and every point in X
            distances[:, k] = np.sum(np.power(self.X - self.means[k], 2), axis=1) ** 0.5
        self.distances = distances

    def __update_assignments(self):
        self.__getdistances()
        # self.assignments is a N x 1 vector, where self.assignments_n is the number of the cluster corresponding to point n
        self.assignments = np.argmin(self.distances, axis=1)

    # X is a (N x 784) array since the dimension of each image is 28x28
    def fit(self, X):
        self.X = X
        self.objective_vals = []
        self.__initialize_means()
        # Run k-Means EM algorithm 10 times...
        for i in range(10):
            self.__update_assignments()
            self.objective_vals.append(self.objective())
            # print('Epoch:', i, 'Objective:', self.objective())
            self.__update_means()

    # This should plot the objective as a function of iteration and verify that it never increases.
    # This assumes that fit() has already been called.
    def plot_verify_objective(self):
        if not self.objective_vals:
            print('call fit() first!')
            return
        assert(np.all(np.array(self.objective_vals[1:]) - np.array(self.objective_vals[:-1]) <= 0))
        plt.figure()
        plt.plot(np.arange(len(self.objective_vals)), self.objective_vals)
        plt.title('K Means Objective, K =' + str(self.K))
        plt.show()

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        return self.means

KMeansClassifier = KMeans(10)
KMeansClassifier.fit(mnist_pics)
print("Final Objective:", KMeansClassifier.objective()/(mnist_pics.shape[0]))

