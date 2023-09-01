#Melissa Aurora Fadanelli Ordaz | A01749483

#Importación de lobrerías
import numpy as np
from collections import Counter

#Calcular la distancia euclidiana
def euc_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

#Se define la clase de knn con sus métodos
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self,x):
        #Calcular la distancia entre el punto x con los demás
        distances = [euc_distance(x, x_train) for x_train in self.X_train]

        #Sacar los valores de k más cercanos
        k_indexes = np.argsort(distances)[:self.k]

        #Sacar las etiquetas
        k_nearest_labels = [self.y_train[i] for i in k_indexes]

        #La etiqueta que es más común
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]