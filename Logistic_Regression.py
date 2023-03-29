# Logistic Regression Algoritması İçin Fit Etme ve Karmaşıklık Matrisi Hesaplama

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

# Veri setinin yüklenmesi ve ön işleme adımlarının gerçekleştirilmesi

df = pd.read_csv('bikeshare.csv')

X = df.drop('target', axis=1) # features

y = df['target'] # target variable

# Veri setinin eğitim ve test kümelerine ayrılması

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression algoritmasının belirlenmesi ve model tanımlanması

lr_classifier = LogisticRegression(random_state=0)

# Modelin eğitim verileri kullanılarak eğitilmesi

lr_classifier.fit(X_train, y_train)

# Test verilerinin model üzerinde çalıştırılması ve tahminlerin elde edilmesi

y_pred_lr = lr_classifier.predict(X_test)

# Karmaşıklık matrisinin hesaplanması

cm_lr = confusion_matrix(y_test, y_pred_lr)

print("Logistic Regression Algoritması Karmaşıklık Matrisi:\n", cm_lr)

"""

[[3983  888]

 [ 858 6556]]
 
 """