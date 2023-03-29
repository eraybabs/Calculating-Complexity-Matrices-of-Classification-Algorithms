# K-Nearest Neighbors Algoritması İçin Fit Etme ve Karmaşıklık Matrisi Hesaplama

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

# Veri setinin yüklenmesi ve ön işleme adımlarının gerçekleştirilmesi

df = pd.read_csv('weatherHistory.csv')

X = df.drop('target', axis=1) # features

y = df['target'] # target variable

# Veri setinin eğitim ve test kümelerine ayrılması

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# K-Nearest Neighbors algoritmasının belirlenmesi ve model tanımlanması

knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Modelin eğitim verileri kullanılarak eğitilmesi

knn_classifier.fit(X_train, y_train)

# Test verilerinin model üzerinde çalıştırılması ve tahminlerin elde edilmesi

y_pred_knn = knn_classifier.predict(X_test)

# Karmaşıklık matrisinin hesaplanması

cm_knn = confusion_matrix(y_test, y_pred_knn)

print("K-Nearest Neighbors Algoritması Karmaşıklık Matrisi:\n", cm_knn)

"""

[[4442  429]

 [2765 3649]]

"""