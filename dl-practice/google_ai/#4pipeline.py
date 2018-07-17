# import a dataset
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# from sklearn import tree
# my_classfier = tree.DecisionTreeClassifier()

from sklearn.neighbors import KNeighborsClassifier
my_classfier = KNeighborsClassifier()

my_classfier.fit(X_train, y_train)

preditions = my_classfier.predict(X_test)
print(preditions)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, preditions))