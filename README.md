# Réseaux Neuronaux Convolutifs (CNN)

Ce projet regroupe plusieurs implémentations de réseaux neuronaux convolutifs (CNN) réalisées avec PyTorch, couvrant trois applications distinctes de la vision par ordinateur.

## Contenu du projet

### 1. Reconnaissance de chiffres manuscrits (`reco_chiffre.ipynb`)

Implémentation d'un CNN pour la reconnaissance de chiffres manuscrits à partir du jeu de données **MNIST**.

- **Architecture** : deux couches de convolution (`Conv2d`) suivies de couches entièrement connectées
- **Optimiseur** : SGD (Stochastic Gradient Descent)
- **Fonction de perte** : Cross-Entropy Loss
- **Entraînement** : 20 époques
- Le modèle entraîné est sauvegardé dans `mnist_cnn.pt`

### 2. Détection de pneumonie (`reco_pneumonie.ipynb`)

CNN binaire permettant de classifier des radiographies thoraciques en deux catégories : **Sain** ou **Pneumonie**.

- **Dataset** : `Ewakaa/pneumonia_classification_chest_xray` (chargé via Hugging Face `datasets`)
- **Architecture** : plusieurs couches de convolution avec max pooling, suivies de couches denses
- **Optimiseur** : Adam
- **Fonction de perte** : Binary Cross-Entropy Loss (BCE)
- **Entraînement** : 15 époques
- Inclut une analyse de la courbe ROC pour évaluer les performances du modèle
- Le modèle entraîné est sauvegardé dans `model_pneumonie.pth`

### 3. Détection de visages (`reco_visage.ipynb`)

Détection de visages dans une image en utilisant deux approches complémentaires :

- **MTCNN** (`mtcnn`) : détecteur de visages basé sur un réseau de neurones
- **Haar Cascades** (OpenCV `CascadeClassifier`) : détecteur classique basé sur des caractéristiques Haar

## Prérequis

```bash
pip install torch torchvision
pip install datasets
pip install mtcnn opencv-python
pip install matplotlib plotly scikit-learn
```

## Structure du projet

```
├── reco_chiffre.ipynb       # Reconnaissance de chiffres (MNIST)
├── reco_pneumonie.ipynb     # Détection de pneumonie (radiographies)
├── reco_visage.ipynb        # Détection de visages
├── étude IA convo.pdf       # Rapport d'étude sur les CNN
└── README.md
```
