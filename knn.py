#Melissa Aurora Fadanelli Ordaz | A01749483

#Importación de librerías
import numpy as np
from collections import Counter
"""
Función para calcular la distancia euclidiana entre dos puntos
Entradas:
    x1: Un punto en el espacio multidimensional (numpy array).
    x2: Otro punto en el espacio multidimensional (numpy array).
Salida:
    distance: La distancia euclidiana entre x1 y x2 (float).
"""
def euc_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

#Se define la clase de knn
class KNN:
    """
    Constructor de la clase
    Entradas:
        k: Número de vecinos cercanos (valor predeterminado = 3).
    """
    def __init__(self, k=3):
        self.k = k

    """
    Método para ajustar el modelo k-NN a los datos de entrenamiento
    Entradas:
        X: Datos de entrenamiento (numpy array de forma [n_samples, n_features]).
        y: Etiquetas correspondientes a los datos de entrenamiento (numpy array de forma [n_samples]).
    """
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    """
    Método para realizar predicciones sobre nuevos datos
    Entradas:
        X: Datos de prueba (numpy array de forma [n_samples, n_features]).
    Salida:
        predictions: Las etiquetas predichas para los datos de prueba (lista de int).
    """
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    """
    Método para realizar una predicción sobre un dato
    Entradas:
        x: Un dato de prueba (numpy array de forma [n_features]).
    Salida:
        label: La etiqueta predicha para el dato de prueba (int).
    """
    def _predict(self,x):
        # Calcular la distancia entre el punto x y los demás puntos de entrenamiento
        distances = [euc_distance(x, x_train) for x_train in self.X_train]

        # Obtener los índices de los k vecinos más cercanos
        k_indexes = np.argsort(distances)[:self.k]

        # Obtener las etiquetas de los k vecinos más cercanos
        k_nearest_labels = [self.y_train[i] for i in k_indexes]

        # Encontrar la etiqueta más común entre los vecinos más cercanos
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
