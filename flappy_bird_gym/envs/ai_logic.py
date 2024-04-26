class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []

def create_gri!d(initial_state, depth):
    """
    Crée une grille (arbre) des mouvements possibles depuis l'état initial jusqu'à une certaine profondeur.
    Chaque nœud représente un état possible et chaque lien une action possible.
    
    :param initial_state: L'état initial du jeu.
    :param depth: La profondeur maximale de l'arbre.
    :return: La racine de l'arbre des mouvements possibles.
    """
    # Fonction récursive pour explorer les états
    def explore_state(current_node, current_depth):
        if current_depth == 0:
            return
        # Pour cet exemple, on suppose deux actions possibles : 1 pour sauter, 0 pour ne rien faire
        for action in [1, 0]:
            # Calculer le nouvel état basé sur l'action (ici simplifié)
            # Vous auriez besoin de votre logique de jeu pour calculer cela correctement
            new_state = (current_node.state[0] + action, current_node.state[1] + 1) # Simplification
            new_node = Node(new_state, current_node, action)
            current_node.children.append(new_node)
            explore_state(new_node, current_depth - 1)
    
    root = Node(initial_state)
    explore_state(root, depth)
    return root

# Exemple d'utilisation avec un état initial simplifié (position, temps) et une profondeur de 3
create_grid((0, 0), 3)
