import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FlappyBirdCNN(nn.Module):
    def __init__(self):
        super(FlappyBirdCNN, self).__init__()
        # Première couche convolutionnelle prenant une image 3 canaux en entrée
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        # Seconde couche convolutionnelle
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        # Une couche de pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Dimensions de l'image après les couches de convolutions et pooling
        # Cela dépend de la taille de votre image d'entrée
        # Pour une image de 512x512, après 2 max pooling, cela devient 128x128
        # 32 est le nombre de sorties de la dernière couche de convolution
        self.size_after_conv = 32 * 128 * 128

        # Couche de connexion dense (fully connected)
        self.fc1 = nn.Linear(self.size_after_conv + 1, 120)  # +1 pour inclure le score comme caractéristique
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x, score):
        # Passage de l'image à travers les couches de convolution et pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Aplatir les données pour la couche fully connected
        x = x.view(-1, self.size_after_conv)
        # Inclure le score comme caractéristique supplémentaire
        x = torch.cat((x, score.unsqueeze(1)), dim=1)
        # Passage à travers les couches fully connected
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def decide(self, image, score):
        # Conversion de l'image numpy RGB en tensor PyTorch et normalisation
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        # Conversion du score en tensor PyTorch
        score_tensor = torch.tensor([score]).float()
        # Obtenir la prédiction du modèle
        output = self.forward(image_tensor, score_tensor)
        # Décider de sauter (1) ou ne rien faire (0) basé sur le seuil de 0.5
        action = 1 if output.item() > 0.5 else 0
        return action


