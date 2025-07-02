# 🧠 Classification des images TID2013 avec CNN / TID2013 Image Classification with CNN

Ce projet vise à entraîner un modèle de réseau de neurones convolutifs (CNN) pour classer les images de la base **TID2013** en deux catégories :  
- `reference` : images originales non altérées  
- `distorted` : images dégradées par divers types de distorsions  

This project aims to train a Convolutional Neural Network (CNN) model to classify images from the **TID2013** dataset into two categories:  
- `reference`: original, unaltered images  
- `distorted`: degraded/distorted images due to various types of alterations  

---

## 📂 Dataset – TID2013

Le jeu de données [TID2013](http://www.ponomarenko.info/tid2013.htm) est une base de référence en qualité d’image.  
Il contient :
- 25 **images de référence**
- 3000 **images distordues** générées à partir des images de référence avec différents niveaux et types de distorsions.

This dataset is commonly used in image quality assessment tasks. It includes:
- 25 **reference images**
- 3000 **distorted images** created by applying various types of noise and distortions.

---

## ⚙️ Étapes du projet / Project Pipeline

### 📦 1. Importation des bibliothèques  
Utilisation de TensorFlow/Keras, NumPy, scikit-learn et Matplotlib.

### 📁 2. Chargement et préparation des données  
- Chargement des images depuis Google Drive  
- Redimensionnement à 224x224  
- Normalisation des pixels

### 🏷️ 3. Encodage des étiquettes  
- Transformation des étiquettes `reference` et `distorted` avec `LabelEncoder`  
- Encodage one-hot pour l’entraînement

### ⚖️ 4. Gestion du déséquilibre des classes  
Le jeu de données est **fortement déséquilibré** (25 vs 3000), un poids de classe a été calculé avec `compute_class_weight`.

### 🧠 5. Création et entraînement du CNN  
- 3 blocs Conv2D + MaxPooling  
- Flatten → Dense(128) → Dense(2 avec softmax)  
- Entraînement pendant 10 époques  
- Sauvegarde du modèle `.keras` et de l’historique `.pkl`

### 🔍 6. Prédiction sur une image test  
Une image de test est prétraitée puis soumise au modèle pour prédiction.

---

## 📉 Résultat observé / Observed Result

Lors de la prédiction sur une image située dans le dossier `reference_images`,  
le modèle a **prédit à tort qu’elle appartenait à la classe `distorted`**.

> 🛑 Cela montre que le modèle a été **biaisé par le déséquilibre des données**,  
> ayant vu très peu d’images `reference` durant l’apprentissage.

---

## 📌 Exemple de prédiction incorrecte

```python
img_path = '/content/drive/My Drive/TID2013/reference_images/I11.jpg'
prepared_image = prepare_image(img_path)
predictions = model.predict(prepared_image)
predicted_label = le.classes_[np.argmax(predictions)]
print(f"Classe prédite : {predicted_label}")  # Résultat : distorted (faux)
