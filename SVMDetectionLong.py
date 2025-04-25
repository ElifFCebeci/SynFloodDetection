import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

print("Kod başlatıldı...")

# Veri Setini Okuma
df = pd.read_csv("Syn.csv", low_memory=False)
print("Veri seti başarıyla okundu.")

# Eksik veya Sonsuz Değerleri Temizleme
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)  # Eksik verileri tamamen kaldır
print("🧹 Eksik veya sonsuz değerler temizlendi.")

# Etiket (Label) ve Özellikleri Ayırma
y = df[" Label"]  # Saldırı/Normal ayrımını içeren sütun
X = df.drop(["Flow ID", "Source IP", "Destination IP", "Timestamp", " Label"], axis=1, errors='ignore')
print("Etiket ve özellikler ayrıldı.")

# Sayısal Olmayan Verileri Çıkar
X = X.select_dtypes(include=[np.number])
print("🔢 Sadece sayısal veriler seçildi.")

# Korelasyonu Yüksek Özellikleri Çıkar (Pearson ile)
corr_matrix = X.corr(method='pearson')
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col].abs() > 0.8)]
X = X.drop(columns=to_drop)
print("Yüksek korelasyona sahip özellikler çıkarıldı.")

# X ve y’nin Boyutlarını Eşitle
df_cleaned = df.drop(columns=to_drop, errors='ignore')  # Yüksek korelasyonlu sütunları y’den de çıkar
y = df_cleaned[" Label"]  # Güncellenmiş y değeri
print("X ve y boyutları eşitlendi.")

# Veri Ölçekleme (SVM için önemli)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Veri ölçeklendi.")

# Eğitim ve Test Setine Ayırma (%50 eğitim, %50 test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)
print("Veri eğitim ve test setlerine ayrıldı.")

# SVM Modeli (RBF Kernel)
model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)
print("SVM modeli eğitildi.")

# Model Testi ve Performans Analizi
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("Model test edildi.")

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# ROC Eğrisi Çizme
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label='SVM ROC curve (RBF Kernel)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVM (RBF Kernel)')
plt.legend()
plt.show()
print("ROC eğrisi çizildi.")