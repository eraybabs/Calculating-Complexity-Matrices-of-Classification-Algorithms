# Random Forest Algoritması İçin Fit Etme ve Karmaşıklık Matrisi Hesaplama

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix


# Veri setinin yüklenmesi ve ön işleme adımlarının gerçekleştirilmesi

df = pd.read_csv('CO2_data.csv')

X = df.drop('target', axis=1) # features

y = df['target'] # target variable

# Veri setinin eğitim ve test kümelerine ayrılması

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest algoritmasının belirlenmesi ve model tanımlanması

rf_classifier = RandomForestClassifier(n_estimators=100)

# Modelin eğitim verileri kullanılarak eğitilmesi

rf_classifier.fit(X_train, y_train)

# Test verilerinin model üzerinde çalıştırılması ve tahminlerin elde edilmesi

y_pred_rf = rf_classifier.predict(X_test)

# Karmaşıklık matrisinin hesaplanması

cm_rf = confusion_matrix(y_test, y_pred_rf)

print("Random Forest Algoritması Karmaşıklık Matrisi:\n", cm_rf)


"""

[[4564  307]

 [ 651 6763]]

"""