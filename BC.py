from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
import matplotlib.pyplot as plt

# train_test_split()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

fig, subaxes = plt.subplots(1, 1, figsize=(6, 6))
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)
print('Breast cancer dataset (learning rate = 0.1,max depth = 3)')
print('Accuracy of GBDT classifier on training set{:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of GBDT on testing set'.format(clf.score(X_test, y_test)))
clf = GradientBoostingClassifier(learning_rate=0.01, max_depth=2, random_state=0)
clf.fit(X_train,y_train)
print('Breast cancer dataset (learning rate = 0.1,max depth = 3)')
print('Accuracy of GBDT classifier on training set{:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of GBDT on testing set'.format(clf.score(X_test, y_test)))
