#Melissa Aurora Fadanelli Ordaz | A01749483
# Importación de bibliotecas
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

# Importación de la implementación de k-NN
from knn import KNN

# Cargar el conjunto de datos Breast Cancer Wisconsin desde sklearn
cancer = datasets.load_breast_cancer()
X, y = cancer.data, cancer.target

# División del conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Creación de una instancia de la implementación de k-NN con k=5
clf = KNN(k=5)

# Ajuste del modelo k-NN a los datos de entrenamiento
clf.fit(X_train, y_train)

# Realización de predicciones en los datos de prueba
predictions = clf.predict(X_test)

# Cálculo de la precisión del modelo
acc = np.sum(predictions == y_test) / len(y_test)
print("Precisión del modelo:", acc)

# Cálculo de la matriz de confusión
confusion = confusion_matrix(y_test, predictions)
print("Matriz de Confusión:")
print(confusion)

# Cálculo del puntaje F1
f1 = f1_score(y_test, predictions)
print("Puntaje F1:", f1)