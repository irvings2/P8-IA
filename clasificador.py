import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#AI

# Cargar datos de entrenamiento
train_data = pd.read_csv('train.csv')

# Entrenamiento con petallength
X_train = train_data['petallength'].values.reshape(-1, 1)
y_train = train_data['class']

clf_length = DecisionTreeClassifier()
clf_length.fit(X_train, y_train)

# Entrenamiento con petalwidth
X_train = train_data['petalwidth'].values.reshape(-1, 1)
y_train = train_data['class']

clf_width = DecisionTreeClassifier()
clf_width.fit(X_train, y_train)

# Cargar datos de prueba
test_data = pd.read_csv('test.csv')

# Clasificar con petallength
X_test = test_data['petallength'].values.reshape(-1, 1)
y_test = test_data['class']

y_pred_length = clf_length.predict(X_test)

# Contar aciertos
count_correct_length = sum(y_pred_length == y_test)
print(f'Clasificador con petallength: {count_correct_length} de {len(y_test)} patrones clasificados correctamente')

# Clasificar con petalwidth
X_test = test_data['petalwidth'].values.reshape(-1, 1)

y_pred_width = clf_width.predict(X_test)

# Contar aciertos
count_correct_width = sum(y_pred_width == y_test)
print(f'Clasificador con petalwidth: {count_correct_width} de {len(y_test)} patrones clasificados correctamente')
