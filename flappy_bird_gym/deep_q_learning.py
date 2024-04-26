import flappy_bird_gym
import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# Hyperparameters
learning_rate = 0.01
gamma = 0.9
buffer_limit = 50000
batch_size = 320

class ReplayBuffer():
    def __init__(self):
        # Initialise un deque pour stocker les etats successifs rencontrées en jeu
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n): # Prend n transitions aleatoires dans la memoire
        mini_batch = random.sample(self.buffer, n)
        observation_old_lst, action_lst, reward_lst, observation_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            observation_old, action, reward, observation, done_mask = transition
            observation_old_lst.append(observation_old)
            action_lst.append([action])
            reward_lst.append([reward])
            observation_lst.append(observation)
            done_mask_lst.append([done_mask])

        return torch.tensor(np.array(observation_old_lst), dtype=torch.float), \
                torch.tensor(np.array(action_lst), dtype=torch.int64), \
                torch.tensor(np.array(reward_lst), dtype=torch.float), \
                torch.tensor(np.array(observation_lst), dtype=torch.float), \
                torch.tensor(np.array(done_mask_lst), dtype=torch.float)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(2, 5)  # Assume input dimension is 2 (horizontal and vertical distances)
        self.fc2 = nn.Linear(5, 2)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if False:
            return random.randint(0, 1)
        else:
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def get_reward(done, info):
    # Initialize reward
    reward = 0

    # Constants for rewards and penalties
    SURVIVAL_REWARD = 15
    DEATH_PENALTY = -1000
    
    # Check if the game is over (bird has died)
    if done:
        reward += DEATH_PENALTY
    else:
        # Add a survival reward for each frame the bird stays alive
        reward += SURVIVAL_REWARD

    return reward



def main():
    env = flappy_bird_gym.make("FlappyBird-v0")
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    print_interval = 100
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    epsilon_start = 0.1
    epsilon_final = 0.01
    epsilon_decay = 2000
    steps_done = 0
    episodes = 1000
    avg_score = 0.0
    for steps_done in range(episodes):
        epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * steps_done / epsilon_decay)
        observation_old = env.reset()
        observation_old = np.array([observation_old[1]+0.0244140625, env._game.player_vel_y]) # On ajoute la vitesse du joueur a l'etat
        score = 0.0
        done = False
        

        while not done: # Tant que l'agnet n'a pas perdu on joue  
            action = q.sample_action(torch.from_numpy(np.array(observation_old)).float(), epsilon) # On demande a l'agent de choisir une action
            observation, reward, done, info = env.step(action) # On execute l'action et on recupere le nouvel etat, la recompense et si l'agent a perdu
            reward = get_reward(done, info) # On calcule la recompense
            score = info['score']
            observation = np.array([observation[1]+0.0244140625, env._game.player_vel_y])
            done_mask = 0.0 if done else 1.0
            memory.put((observation_old, action, reward, observation, done_mask)) # On ajoute la transition dans la memoire
            observation_old = observation # On retient l'etat precedent

            if done: # Si l'agent a perdu on sort de la boucle
                break
        avg_score += score
            
        if memory.size() > 2000: # Si la memoire est assez grande on entraine le reseau
            train(q, q_target, memory, optimizer)
        steps_done += 1 # On teint compte du nombre de parties jouees

        if steps_done % print_interval == 0:
            avg_score = avg_score / print_interval 
            print("Steps: {}, Score: {:.1f}, Buffer Size: {}, Eps: {:.2f}%".format(steps_done, avg_score, memory.size(), epsilon * 100))
            avg_score = 0.0

    obs = env.reset()
    obs = np.array([obs[1], env._game.player_vel_y])
    while True:
        # Decide next action with A* search
        action = q.sample_action(torch.from_numpy(np.array(observation_old)).float(), epsilon)

        # Execute chosen action
        obs, reward, done, info = env.step(action)
        obs = np.array([obs[1], env._game.player_vel_y])
        print("Plyer_vel {0}",env._game.player_vel_y)
        # Render the game
        env.render()
        time.sleep(1 / 30)  # Adjust FPS for human-friendly viewing

        # Break if the game is over
        if done:
            break
    env.close()

if __name__ == '__main__':
    main()
