#Melissa Aurora Fadanelli Ordaz | A01749483
#Importar bibliotecas
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, f1_score

# Importación de la implementación de k-NN
from knn import KNN

# Creación de un mapa de colores para la gráfica de dispersión
cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

# Cargar el conjunto de datos Iris desde sklearn
iris = datasets.load_iris()
X, y = iris.data, iris.target

# División del conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Creación y visualización de una gráfica de dispersión de los datos
plt.figure()
plt.scatter(X[:,2] , X[:,3] , c = y , cmap=cmap , edgecolor='k', s=20)
plt.show()

k_values = [5,12,26,35]
for i in range(len(k_values)):
    # Creación de una instancia de la implementación de k-NN con k=5
    clf = KNN(k=k_values[i])

    # Ajuste del modelo k-NN a los datos de entrenamiento
    clf.fit(X_train, y_train)

    # Realización de predicciones en los datos de prueba
    predictions = clf.predict(X_test)

    # Impresión de las predicciones
    print("Predicciones:")
    print(predictions)

    # Cálculo de la precisión del modelo
    acc = np.sum(predictions == y_test) / len(y_test)
    print(acc)

    # Calcular la matriz de confusión
    confusion = confusion_matrix(y_test, predictions)
    print("Matriz de Confusión:")
    print(confusion)

    # Calcular el puntaje F1
    f1 = f1_score(y_test, predictions, average='weighted')
    print("Puntaje F1:", f1)