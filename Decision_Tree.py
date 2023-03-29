# Decision Tree Algoritması İçin Fit Etme ve Karmaşıklık Matrisi Hesaplama

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

# Veri setinin yüklenmesi ve ön işleme adımlarının gerçekleştirilmesi

df = pd.read_csv('headbrain.csv')

X = df.drop('target', axis=1) # features

y = df['target'] # target variable

# Veri setinin eğitim ve test kümelerine ayrılması

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree algoritmasının belirlenmesi ve model tanımlanması

dt_classifier = DecisionTreeClassifier()

# Modelin eğitim verileri kullanılarak eğitilmesi

dt_classifier.fit(X_train, y_train)

# Test verilerinin model üzerinde çalıştırılması ve tahminlerin elde edilmesi

y_pred_dt = dt_classifier.predict(X_test)

# Karmaşıklık matrisinin hesaplanması

cm_dt = confusion_matrix(y_test, y_pred_dt)

print("Decision Tree Algoritması Karmaşıklık Matrisi:\n", cm_dt)

"""

[[4464  407]

 [ 677 6737]]

 """