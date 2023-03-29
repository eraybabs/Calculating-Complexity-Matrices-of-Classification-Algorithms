# Naive Bayes Algoritması İçin Fit Etme ve Karmaşıklık Matrisi Hesaplama

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix

# Veri setinin yüklenmesi ve ön işleme adımlarının gerçekleştirilmesi

df = pd.read_csv('polynomial_regression.csv')

X = df.drop('target', axis=1) # features

y = df['target'] # target variable

# Veri setinin eğitim ve test kümelerine ayrılması

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes algoritmasının belirlenmesi ve model tanımlanması

nb_classifier = GaussianNB()

# Modelin eğitim verileri kullanılarak eğitilmesi

nb_classifier.fit(X_train, y_train)

# Test verilerinin model üzerinde çalıştırılması ve tahminlerin elde edilmesi

y_pred_nb = nb_classifier.predict(X_test)

# Karmaşıklık matrisinin hesaplanması

cm_nb = confusion_matrix(y_test, y_pred_nb)

print("Naive Bayes Algoritması Karmaşıklık Matrisi:\n", cm_nb)

"""

[[4528  343]

 [2391 4023]]

"""