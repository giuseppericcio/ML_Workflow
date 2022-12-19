import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

## Read Dataset
df = pd.read_csv('./Dataset/Train.csv')

print(df['Class'].value_counts())

x = df.drop(columns = ['ID','Class'], axis=1)
y = df['Class']

## Split the Dataset into train and test
trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)

sc=StandardScaler()

scaler = sc.fit(trainX)
trainX_scaled = scaler.transform(trainX)
testX_scaled = scaler.transform(testX)

## MLP Classifier
mlp_clf = MLPClassifier(hidden_layer_sizes=(20,15,10),
                        max_iter = 1000,activation = 'relu',
                        solver = 'sgd')

mlp_clf.fit(trainX_scaled, trainY)

## Model Evaluation
y_pred = mlp_clf.predict(testX_scaled)

print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))

fig = plot_confusion_matrix(mlp_clf, testX_scaled, testY, display_labels=mlp_clf.classes_)
fig.figure_.suptitle("Confusion Matrix for AGB Dataset")
plt.show()

print(classification_report(testY, y_pred))

plt.plot(mlp_clf.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

## Hyperparameter Tuning
param_grid = {
    'hidden_layer_sizes': [(20,15,10)],
    'max_iter': [500, 800, 1000],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

grid = GridSearchCV(mlp_clf, param_grid, n_jobs= -1, cv=5)
grid.fit(trainX_scaled, trainY)

print(grid.best_params_)

grid_predictions = grid.predict(testX_scaled)

print('Accuracy: {:.2f}'.format(accuracy_score(testY, grid_predictions)))