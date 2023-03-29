# MLP Classification Algoritması İçin Fit Etme ve Karmaşıklık Matrisi Hesaplama

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix

# Veri setinin yüklenmesi ve ön işleme adımlarının gerçekleştirilmesi

df = pd.read_csv('CO2_data.csv')

X = df.drop('target', axis=1) # features

y = df['target'] # target variable

# Veri setinin eğitim ve test kümelerine ayrılması

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLP Classification algoritmasının belirlenmesi ve model tanımlanması

mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=0)

# Modelin eğitim verileri kullanılarak eğitilmesi

mlp_classifier.fit(X_train, y_train)

# Test verilerinin model üzerinde çalıştırılması ve tahminlerin elde edilmesi

y_pred_mlp = mlp_classifier.predict(X_test)

# Karmaşıklık matrisinin hesaplanması

cm_mlp = confusion_matrix(y_test, y_pred_mlp)

print("MLP Classification Algoritması Karmaşıklık Matrisi:\n", cm_mlp)

"""

[[4262  609]

 [ 619 6795]]

"""