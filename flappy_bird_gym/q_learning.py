import numpy as np
import random
import flappy_bird_gym
import time

# Environment setup
env = flappy_bird_gym.make("FlappyBird-v0")

# Q-table dimensions
# We discretize H_dist and V_dist into bins
num_h_dist_bins = 10  # Number of bins for horizontal distance
num_v_dist_bins = 10  # Number of bins for vertical distance
num_actions = 2       # Actions: 0: do nothing, 1: flap

# Initialize Q-table with zeros
Q = np.zeros((num_h_dist_bins, num_v_dist_bins, num_actions))

# Learning parameters
learning_rate = 0.0001    #0.0001 is good
gamma = 0.9  # Discount factor for future rewards
epsilon = 0.1  # Exploration factor

def get_reward(done, info):
    """
    Calculate the reward for the agent's next state.
    
    :param next_obs: The next observation from the environment after taking an action
    :param done: Boolean indicating if the episode has ended
    :param info: A dictionary containing additional data from the environment (e.g., score)
    """
    # Initialize reward
    reward = 0

    # Constants for rewards and penalties
    SURVIVAL_REWARD = 15
    DEATH_PENALTY = -1000
    PIPE_PASS_REWARD = 100
    
    # Check if the game is over (bird has died)
    if done:
        reward += DEATH_PENALTY
    else:
        # Add a survival reward for each frame the bird stays alive
        reward += SURVIVAL_REWARD

    return reward


def discretize_state(observation):
    """ Convert continuous state into discrete bins for Q-table. """
    h_dist, v_dist = observation
    v_dist += 0.0244140625
    # Discretizing each dimension based on observed practical ranges
    # Assume max horizontal distance observable is 300 pixels, vertical range around 200 pixels
    discrete_h_dist = int(h_dist / 30)  # Assuming the max horizontal distance is around 300
    discrete_v_dist = int((v_dist + 100) / 20)  # Normalize and scale vertical distance

    # Make sure the bins are within the range of the Q-table
    discrete_h_dist = min(discrete_h_dist, num_h_dist_bins - 1)
    discrete_v_dist = min(discrete_v_dist, num_v_dist_bins - 1)

    return discrete_h_dist, discrete_v_dist

def choose_action(state):
    """ Epsilon-greedy action selection """
    if False:#random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Random action (flap or not)
    else:
        return np.argmax(Q[state])  # Best action from Q-table

def update_q_table(state, action, reward, next_state):
    """ Update Q-value by Q-learning formula """
    best_next_action = np.argmax(Q[next_state])  # Best next action
    td_target = reward + gamma * Q[next_state][best_next_action]
    td_error = td_target - Q[state][action]
    Q[state][action] += learning_rate * td_error


# Main loop
episodes = 500
print_interval = 50
avg_score = 0.0
for steps_done in range(episodes):
    done = False
    obs = env.reset()
    current_state = discretize_state(obs)
    score = 0.0

    while not done:
        action = choose_action(current_state)
        next_obs, reward, done, info = env.step(action)
        score += info['score']
        reward = get_reward(done, info)
        next_state = discretize_state(next_obs)
        update_q_table(current_state, action, reward, next_state)
        current_state = next_state
    avg_score += score
    
    if steps_done % print_interval == 0:
            avg_score = avg_score / print_interval 
            print("Steps: {}, Score: {:.1f}, Eps: {:.2f}% ".format(steps_done, avg_score, epsilon * 100))
            avg_score = 0.0

    # Optionally, reduce epsilon (exploration factor) gradually

obs = env.reset()
score = 0.0
current_state = discretize_state(obs)
done = False
while not done:
        action = choose_action(current_state)
        next_obs, reward, done, info = env.step(action)
        reward = get_reward(done, info)
        score = reward
        env.render()
        time.sleep(1 / 120)
        next_state = discretize_state(next_obs)
        current_state = next_state

env.close()  # Close the environment after training