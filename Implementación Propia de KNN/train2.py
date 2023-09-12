#Melissa Aurora Fadanelli Ordaz | A01749483
#Importar bibliotecas
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Importación de la implementación de k-NN
from knn import KNN

# Creación de un mapa de colores para la gráfica de dispersión
cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

# Cargar el conjunto de datos Wine desde sklearn
wine = datasets.load_wine()
X, y = wine.data, wine.target

# División del conjunto de datos en un conjunto temporal y prueba
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.4, random_state=1234)

#Usando el conjunto temporal podemos dividirlo en dos para el entrenamiento y la validación
X_train, X_validation, y_train, y_validation = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1234)

# Creación y visualización de una gráfica de dispersión de los datos
plt.figure()
plt.scatter(X[:,2] , X[:,3] , c = y , cmap=cmap , edgecolor='k', s=20)
plt.show()

# Visualización de los datos de entrenamiento, validación y prueba
plt.figure(figsize=(12, 4))

# Datos de entrenamiento
plt.subplot(131)
plt.scatter(X_train[:, 2], X_train[:, 3], c=y_train, cmap=cmap, edgecolor='k', s=20)
plt.title('Conjunto de Entrenamiento')
plt.xlabel('Característica 2')
plt.ylabel('Característica 3')

# Datos de validación
plt.subplot(132)
plt.scatter(X_validation[:, 2], X_validation[:, 3], c=y_validation, cmap=cmap, edgecolor='k', s=20)
plt.title('Conjunto de Validación')
plt.xlabel('Característica 2')
plt.ylabel('Característica 3')

# Datos de prueba
plt.subplot(133)
plt.scatter(X_test[:, 2], X_test[:, 3], c=y_test, cmap=cmap, edgecolor='k', s=20)
plt.title('Conjunto de Prueba')
plt.xlabel('Característica 2')
plt.ylabel('Característica 3')

plt.tight_layout()
plt.show()

#k_values = [5,12,26,35]
k_values=[1,3,4,6]

# Listas para almacenar las precisiones en cada conjunto
accuracies_train = []
accuracies_validation = []
accuracies_test = []

for i in range(len(k_values)):
    # Creación de una instancia de la implementación de k-NN con k=5
    clf = KNN(k=k_values[i])

    # Ajuste del modelo k-NN a los datos de entrenamiento
    clf.fit(X_train, y_train)

    #Imprimimos una frase que muestre el valor de k que estamos usando
    print("Usando valor de k={0}\n".format(k_values[i]))

    # Realización de predicciones en los datos de entrenamiento
    predictions_train = clf.predict(X_train)

    # Impresión de las predicciones
    print("Predicciones con datos de entrenamiento:")
    print(predictions_train)

    # Cálculo de la precisión del modelo
    acc_train = np.sum(predictions_train == y_train) / len(y_train)
    accuracies_train.append(acc_train)
    print(acc_train)

    # Calcular la matriz de confusión
    confusion_train = confusion_matrix(y_train, predictions_train)
    print("Matriz de Confusión con datos de entrenamiento:")
    print(confusion_train)

    # Calcular el puntaje F1
    f1_train = f1_score(y_train, predictions_train, average='weighted')
    print("Puntaje F1 con datos de entrenamiento:", f1_train)


    """
    Repetimos los procedimientos anteriores 
    con el conjunto de datos de validación
    """

    # Realización de predicciones en los datos de validación
    predictions_val = clf.predict(X_validation)

    # Impresión de las predicciones
    print("Predicciones con datos de validación:")
    print(predictions_val)

    # Cálculo de la precisión del modelo
    acc_val = np.sum(predictions_val == y_validation) / len(y_validation)
    accuracies_validation.append(acc_val)
    print(acc_val)

    # Calcular la matriz de confusión
    confusion_val = confusion_matrix(y_validation, predictions_val)
    print("Matriz de Confusión con datos de validación:")
    print(confusion_val)

    # Calcular el puntaje F1
    f1_val = f1_score(y_validation, predictions_val, average='weighted')
    print("Puntaje F1 con datos de validación:", f1_val)


    """
    Repetimos los procedimientos anteriores 
    con el conjunto de datos de prueba
    """

    # Realización de predicciones en los datos de prueba
    predictions_test = clf.predict(X_test)

    # Impresión de las predicciones
    print("Predicciones con datos de prueba:")
    print(predictions_test)

    # Cálculo de la precisión del modelo
    acc_test = np.sum(predictions_test == y_test) / len(y_test)
    accuracies_test.append(acc_test)
    print(acc_test)

    # Calcular la matriz de confusión
    confusion_test = confusion_matrix(y_test, predictions_test)
    print("Matriz de Confusión on datos de prueba:")
    print(confusion_test)

    # Calcular el puntaje F1
    f1_test = f1_score(y_test, predictions_test, average='weighted')
    print("Puntaje F1 on datos de prueba:", f1_test)

    print("\n\n")

"""
Código para crear una gráfica comparativa
que utilice las precisiones de los tres conjuntos
"""
# Crear una gráfica comparativa de las precisiones
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies_train, marker='o', label='Entrenamiento')
plt.plot(k_values, accuracies_validation, marker='o', label='Validación')
plt.plot(k_values, accuracies_test, marker='o', label='Prueba')
plt.title('Precisión vs. Valor de k')
plt.xlabel('Valor de k')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.show()

"""
Técnicas de ajsute
"""
# Antes de la normalización
clf.fit(X_train, y_train)
accuracy_train = np.sum(clf.predict(X_train) == y_train) / len(y_train)
accuracy_validation = np.sum(clf.predict(X_validation) == y_validation) / len(y_validation)
accuracy_test = np.sum(clf.predict(X_test) == y_test) / len(y_test)

# Después de la normalización
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

print("Precisiones antes de la normalización:")
print("Entrenamiento:", accuracy_train)
print("Validación:", accuracy_validation)
print("Prueba:", accuracy_test)

clf.fit(X_train_scaled, y_train)
accuracy_train_scaled = np.sum(clf.predict(X_train_scaled) == y_train) / len(y_train)
accuracy_validation_scaled = np.sum(clf.predict(X_validation_scaled) == y_validation) / len(y_validation)
accuracy_test_scaled = np.sum(clf.predict(X_test_scaled) == y_test) / len(y_test)
print("\nPrecisiones después de la normalización:")
print("Entrenamiento:", accuracy_train_scaled)
print("Validación:", accuracy_validation_scaled)
print("Prueba:", accuracy_test_scaled)

# Antes de la selección de características
clf.fit(X_train_scaled, y_train)
accuracy_validation_scaled = np.sum(clf.predict(X_validation_scaled) == y_validation) / len(y_validation)

print("\nPrecisiones antes de la selección de características:")
print("Validación:", accuracy_validation_scaled)

# Después de la selección de características
selector = SelectKBest(score_func=f_classif, k=2)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_validation_selected = selector.transform(X_validation_scaled)
X_test_selected = selector.transform(X_test_scaled)

clf.fit(X_train_selected, y_train)
accuracy_validation_selected = np.sum(clf.predict(X_validation_selected) == y_validation) / len(y_validation)

print("\nPrecisiones después de la selección de características:")
print("Validación con características seleccionadas:", accuracy_validation_selected)

# Antes de la búsqueda de hiperparámetros
clf.fit(X_train_scaled, y_train)
predictions_train_scaled = clf.predict(X_train_scaled)
accuracy_validation_selected = np.sum(predictions_train_scaled == y_validation) / len(y_validation)

# Después de la búsqueda de hiperparámetros
param_grid = {'n_neighbors': [1, 3, 5, 7, 9]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train_selected, y_train)

best_k = grid_search.best_params_['n_neighbors']
clf = KNeighborsClassifier(n_neighbors=best_k)
clf.fit(X_train_selected, y_train)
accuracy_validation_hyperparam = np.sum(clf.predict(X_validation_selected) == y_validation) / len(y_validation)

print("\nPrecisiones antes de la búsqueda de hiperparámetros:")
print("Validación con características seleccionadas:", accuracy_validation_selected)

print("\nPrecisiones después de la búsqueda de hiperparámetros:")
print(f"Validación con k óptimo ({best_k}):", accuracy_validation_hyperparam)