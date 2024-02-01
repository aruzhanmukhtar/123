import numpy as np  # Добавляем эту строку для импорта NumPy
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Загрузка данных
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели логистической регрессии с помощью NumPy
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, weights):
    return sigmoid(np.dot(X, weights))

def logistic_regression(X, y, learning_rate, epochs):
    X = np.c_[np.ones(X.shape[0]), X]
    weights = np.zeros(X.shape[1])

    for epoch in range(epochs):
        predictions = predict(X, weights)
        errors = y - predictions
        gradient = np.dot(X.T, errors)
        weights += learning_rate * gradient

    return weights

learning_rate = 0.01
epochs = 10000
weights = logistic_regression(X_train, y_train, learning_rate, epochs)

# Тестирование модели и расчет метрик
X_test = np.c_[np.ones(X_test.shape[0]), X_test]  # Этот код был вызвал ошибку, теперь исправлен
y_pred = np.round(predict(X_test, weights))

metrics = {
    'Accuracy': accuracy_score,
    'Precision': precision_score,
    'Recall': recall_score,
    'F1 Score': f1_score
}

for metric_name, metric_func in metrics.items():
    if metric_name == 'Accuracy':
        metric_value = metric_func(y_test, y_pred)
    else:
        metric_value = metric_func(y_test, y_pred, average='macro')
    print(f"{metric_name}: {metric_value:.4f}")

# Построение линейной границы решения
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = predict(np.c_[np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel()], weights)
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
plt.title("Linear Decision Boundary")
plt.show()

# Добавление регуляризации (L2 регуляризация)
model_with_regularization = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs')
model_with_regularization.fit(X_train, y_train)

y_pred_regularization = model_with_regularization.predict(X_test)

for metric_name, metric_func in metrics.items():
    if metric_name == 'Accuracy':
        metric_value = metric_func(y_test, y_pred_regularization)
    else:
        metric_value = metric_func(y_test, y_pred_regularization, average='macro')
    print(f"{metric_name} with regularization (L2): {metric_value:.4f}")
