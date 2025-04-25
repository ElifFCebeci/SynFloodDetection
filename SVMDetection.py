import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

print("Kod başlatıldı...")

# Veri Setini Okuma
df = pd.read_csv("Syn.csv", low_memory=False)
print("Veri seti başarıyla okundu.")

# Büyük Veri İçin Örnekleme (Sampling) ile Veri Küçültme
df_sample = df.sample(n=100000, random_state=42) 
print(f"Küçültülmüş veri seti: {df_sample.shape}")

# Eksik veya Sonsuz Değerleri Temizleme
df_sample.replace([np.inf, -np.inf], np.nan, inplace=True)
df_sample.dropna(inplace=True)
print("Eksik veya sonsuz değerler temizlendi.")

# Etiket (Label) ve Özellikleri Ayırma
y = df_sample[" Label"].astype(str)
X = df_sample.drop(["Flow ID", "Source IP", "Destination IP", "Timestamp", " Label"], axis=1, errors='ignore')
print("Etiket ve özellikler ayrıldı.")

# Sayısal Olmayan Verileri Çıkar
X = X.select_dtypes(include=[np.number])
print("Sadece sayısal veriler seçildi.")

# Korelasyonu Yüksek Özellikleri Çıkar (Pearson ile)
corr_matrix = X.corr(method='pearson')
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col].abs() > 0.8)]
X = X.drop(columns=to_drop)
print("Yüksek korelasyona sahip özellikler çıkarıldı.")

# PCA ile Özellik Boyutunu Küçültme
pca = PCA(n_components=30)  # Veri boyutunu 30 bileşene indirerek hızlandır
X_reduced = pca.fit_transform(X)
print(f"PCA sonrası veri boyutu: {X_reduced.shape}")

# 8Etiketleri Sayısala Çevirme
le = LabelEncoder()
y = le.fit_transform(y)
print("Etiketler sayısal hale getirildi.")

# Veri Ölçekleme (SVM için önemli)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reduced)
print("Veri ölçeklendi.")

# Eğitim ve Test Setine Ayırma
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.6, random_state=42)
print(f"Eğitim seti: {X_train.shape[0]} örnek, Test seti: {X_test.shape[0]} örnek")

# Optimizasyonlu SVM Modeli (Linear Kernel)
model = SVC(kernel='linear', probability=True, max_iter=500, tol=1e-2)

try:
    model.fit(X_train, y_train)
    print("SVM modeli başarıyla eğitildi.")
except Exception as e:
    print(f"Model eğitilirken hata oluştu: {e}")

# Model Testi ve Performans Analizi
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("Model test edildi.")

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# ROC Eğrisi Çizme
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label='SVM ROC curve (Linear Kernel)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVM (Linear Kernel)')
plt.legend()
plt.show()
print("ROC eğrisi çizildi.")
