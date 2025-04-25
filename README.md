# SynFloodDetection
Veri Seti (Dataset)
Veri seti, modelin öğrenmesi için kullanılan ham veridir. Bu projede Syn.csv adlı dosya kullanılıyor, içinde ağ trafiği bilgileri bulunuyor.

Veri Temizleme (Data Cleaning)
Veri setinde NaN (eksik değerler) veya sonsuz değerler (Inf) olabilir. Bunları temizlemek, modelin doğru çalışmasını sağlar.
Etiket (Label) ve Özellikler (Features)
- Etiket (Label): Modelin tahmin etmeye çalıştığı sınıftır. Örneğin "SYN_FLOOD" veya "BENIGN" olabilir.
- Özellikler (Features): Modelin tahmin yaparken kullandığı bağımsız değişkenlerdir.

Sayısal Veri Dönüştürme
Model yalnızca sayılarla çalışabilir, bu yüzden metinsel veriler çıkarılır ve tüm girişler sayısal hale getirilir.
Korelasyon (Correlation)
Özellikler arasındaki ilişkiyi ölçer. Eğer iki özellik çok yüksek korelasyona sahipse, biri diğerini gereksiz kılar ve veri setinden çıkarılır.
Veri Ölçekleme (Standardization)
Veri setindeki değişkenler farklı ölçeklerde olabilir (bazıları 0-1 arasında, bazıları 1000'lerde). StandardScaler() kullanarak tüm veriyi dengeli bir ölçeğe getiriyoruz.
Eğitim ve Test Seti (Train-Test Split)
Makine öğrenmesi modelini eğitmek için veriyi ikiye ayırıyoruz:
- Eğitim Seti (Training Set): Modelin öğrendiği veriler.
- Test Seti (Test Set): Modelin doğruluğunu test etmek için kullanılan veriler.

SVM (Support Vector Machine) Modeli
SVM, verileri iki sınıfa ayırmak için en iyi çizgiyi (veya hiper düzlemi) bulan bir makine öğrenmesi algoritmasıdır.
RBF Kernel
SVM'de kullanılan RBF (Radial Basis Function) kernel, karmaşık veri yapılarını daha iyi öğrenebilir. Kıvrımlı bir bölme sağlar.
Classification Report
Modelin doğruluğunu görmek için Precision, Recall, F1-score gibi metrikleri içeren bir performans raporu oluşturulur.
ROC AUC Score
Modelin ne kadar iyi tahmin yaptığını ölçen bir skordur. 1.0’e yakınsa mükemmel, 0.5’e yakınsa rastgele tahmin yapıyor demektir.
ROC Eğrisi
Saldırıyı doğru tahmin etme oranı (True Positive Rate) ile yanlış tahmin oranını (False Positive Rate) gösteren grafik. Modelin performansını görselleştirir.
------------
Dataset
The dataset is the raw data used for model learning. This project uses Syn.csv, which contains network traffic information.
Data Cleaning
The dataset may contain NaN (missing values) or infinite values (Inf). Removing these ensures the model functions correctly.
Labels and Features
- Label: The class that the model attempts to predict. For example, it could be "SYN_FLOOD" or "BENIGN".
- Features: Independent variables used by the model to make predictions.

Numerical Data Conversion
The model can only work with numerical values, so textual data is removed, and all inputs are converted into numerical format.
Correlation
Correlation measures the relationship between features. If two features have high correlation, one may be redundant and is removed from the dataset.
Standardization (Data Scaling)
Variables in the dataset may have different scales (some values range between 0-1, while others are in the thousands).
Using StandardScaler(), we normalize the data to ensure a balanced scale.
Train-Test Split
The machine learning model is trained by splitting the dataset into two parts:
- Training Set: The data the model learns from.
- Test Set: The data used to evaluate the model’s accuracy.

SVM (Support Vector Machine) Model
SVM is a machine learning algorithm that finds the best boundary (or hyperplane) to separate data into two classes.
RBF Kernel
RBF (Radial Basis Function) kernel is used in SVM to learn complex data structures more effectively.
It creates a flexible decision boundary that adapts to the shape of the data.
Classification Report
A performance report that displays metrics like Precision, Recall, and F1-score to evaluate model accuracy.
ROC AUC Score
A score that measures how well the model makes predictions.
- Close to 1.0 → Excellent
- Close to 0.5 → Random predictions

ROC Curve
A graph that visualizes the model’s ability to correctly identify attacks (True Positive Rate) vs. the false alarm rate (False Positive Rate).
