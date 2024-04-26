import torch
import torch.nn as nn
import torch.nn.functional as F

class FlappyBirdAI(nn.Module):
    def __init__(self):
        super(FlappyBirdAI, self).__init__()
        # Définition d'un réseau simple avec une couche cachée
        self.fc1 = nn.Linear(2, 5)  # 2 entrées pour les paramètres, 5 neurones dans la couche cachée
        self.fc2 = nn.Linear(5, 1)  # Sortie de la couche cachée vers 1 neurone de sortie

    def forward(self, x):
        # Passage des données à travers le réseau
        x = F.relu(self.fc1(x))  # Activation ReLU pour la couche cachée
        x = torch.sigmoid(self.fc2(x))  # Activation sigmoid pour obtenir une sortie entre 0 et 1
        return x

    def decide(self, vertical_dist, horizontal_dist):
        # Convertir les entrées en tensor PyTorch et normaliser si nécessaire
        inputs = torch.tensor([vertical_dist, horizontal_dist]).float().unsqueeze(0)
        # Obtenir la prédiction du modèle
        output = self.forward(inputs)
        # Décider de sauter (1) ou ne rien faire (0) basé sur le seuil de 0.5
        action = 1 if output.item() > 0.5 else 0
        return action

# Exemple d'utilisation
model = FlappyBirdAI()
action = model.decide(vertical_dist=100, horizontal_dist=50)  # Exemple de paramètres
print("Action:", action)
