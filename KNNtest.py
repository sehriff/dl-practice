# C:\Users\sunerhan\Documents\机器学习资料\第02课：KNN 算法——不学习我也能预测.PDF
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbours = 15

# import some data to play with
iris = datasets.load_iris()

# we only take the first two features
# slicing by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

h = 0.2 # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier that fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbours, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min,y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap = cmap_light)

    #Plot also the training points
    plt .scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                 edgecolors='k', s=20)
    plt.xlim(xx.min())
    plt.ylim(yy.min())
    plt.title("3-Class classification (k = %i, weight = '%s)" %(n_neighbours, weights))
    
plt.show()