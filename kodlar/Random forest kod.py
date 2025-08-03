import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Grafiklerin stilini ve yazı tipi boyutunu ayarlayalım
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

try:
    # --- 1. VERİ YÜKLEME VE ÖN İŞLEME ---
    df = pd.read_csv(r'C:\Users\Bilge\OneDrive\Masaüstü\476 sunum için gerekli şeyler/duzenlenmis_veri.csv')

    features = [
        'Hane_Basina_ADSL_Abonesi',
        'Toplam_Dogurganlik_Hizi',
        'Kadin_Surucu_Belgesi_Orani',
        'Net_Goc_Hizi',
        'Kisi_Basina_Sosyal_Yardim_Tutari',
        'NUFUS_YOGUNLUGU'
    ]
    target = 'Hedef_Değişken'

    X = df[features]
    y = df[target]

    # Eksik verileri temizle
    if y.isnull().any():
        valid_indices = y.notna()
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]

    if X.isnull().values.any():
        for col in X.columns:
            if X[col].isnull().any():
                mean_value = X[col].mean()
                X.loc[:, col] = X[col].fillna(mean_value)

    # Hedef değişkeni sayısal formata dönüştür
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_

    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

    # --- 2. HİPERPARAMETRE OPTİMİZASYONU (GridSearchCV) ---
    print("Hiperparametre optimizasyonu başlıyor... Bu işlem biraz zaman alabilir.")

    # Aranacak parametreler ve değerleri
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    # GridSearchCV nesnesini oluştur
    # cv=5 -> 5 katlı çapraz doğrulama
    # n_jobs=-1 -> İşlemler için tüm işlemci çekirdeklerini kullan
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                               param_grid=param_grid,
                               cv=5,
                               n_jobs=-1,
                               verbose=0, # Raporlama seviyesi
                               scoring='accuracy')

    # Optimizasyonu başlat
    grid_search.fit(X_train, y_train)

    # En iyi modeli ve parametreleri al
    best_rf_model = grid_search.best_estimator_
    print("\nOptimizasyon tamamlandı.")
    print("="*50)
    print("Bulunan En İyi Hiperparametreler:")
    print(grid_search.best_params_)
    print("="*50)


    # --- 3. EN İYİ MODELİ DEĞERLENDİRME ---
    y_pred = best_rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"\nEn İyi Modelin Doğruluk (Accuracy) Değeri: {accuracy:.4f}\n")
    print("Sınıflandırma Raporu:\n", report)


    # --- 4. GÖRSELLEŞTİRME ---

    # Karışıklık Matrisi (Confusion Matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Karışıklık Matrisi (Confusion Matrix)', fontsize=16)
    plt.ylabel('Gerçek Değerler', fontsize=12)
    plt.xlabel('Tahmin Edilen Değerler', fontsize=12)
    plt.show()


    # Öznitelik Önem Düzeyleri
    feature_importances = pd.DataFrame(best_rf_model.feature_importances_,
                                       index = X_train.columns,
                                       columns=['Önem Düzeyi']).sort_values('Önem Düzeyi', ascending=False)

    plt.figure(figsize=(10, 7))
    sns.barplot(x=feature_importances['Önem Düzeyi'], y=feature_importances.index, palette='viridis')
    plt.title('Öznitelik Önem Düzeyleri', fontsize=16)
    plt.xlabel('Önem Düzeyi', fontsize=12)
    plt.ylabel('Öznitelikler', fontsize=12)
    plt.show()


except FileNotFoundError:
    print("Hata: 'duzenlenmis_veri.csv' dosyası bulunamadı. Lütfen dosyanın doğru yolda olduğundan emin olun.")
except Exception as e:
    print(f"Beklenmedik bir hata oluştu: {e}")
