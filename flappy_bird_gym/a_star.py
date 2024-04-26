import time
import flappy_bird_gym
import numpy as np
import copy

PIPE_WIDTH = 52
PLAYER_WIDTH = 34
PLAYER_HEIGHT = 24
PIPE_HEIGHT = 320
PLAYER_FLAP_ACC = -9
PLAYER_MAX_VEL_Y = 10  
PLAYER_MIN_VEL_Y = -8
PLAYER_ACC_Y = 1 
PIPE_VEL_X = -4

def simulate_action(env, action):
    # Get the current state of the game
    player_y = env._game.player_y

    player_vel_y = env._game.player_vel_y
    base_y = env._game.base_y

    # Here we simulate the action to get the future state
    if action == 1:
            if player_y > -2 * PLAYER_HEIGHT:
                player_vel_y = PLAYER_FLAP_ACC

    if player_vel_y < PLAYER_MAX_VEL_Y and not env._game._player_flapped:
            player_vel_y += PLAYER_ACC_Y
    
    player_y += min(player_vel_y,
                             base_y - player_y - PLAYER_HEIGHT)
    

    # Get the distance to the next pipe
    up_pipe = low_pipe = None
    h_dist = 0
    for up_pipe, low_pipe in zip(env._game.upper_pipes,
                                    env._game.lower_pipes):
        h_dist = (low_pipe["x"] + PIPE_WIDTH / 2
                    - (env._game.player_x - PLAYER_WIDTH / 2))
        h_dist += 3  # extra distance to compensate for the buggy hit-box
        if h_dist >= 0:
            break

    h_dist += PIPE_VEL_X    # Move the pipes to the left

    upper_pipe_y = up_pipe["y"] + PIPE_HEIGHT
    lower_pipe_y = low_pipe["y"]

    v_dist = (upper_pipe_y + lower_pipe_y + 25) / 2 - (player_y
                                                    + PLAYER_HEIGHT/2)
    
    

    if env._normalize_obs:
        h_dist /= env._screen_size[0]
        v_dist /= env._screen_size[1]
        kk /= env._screen_size[1]

    return np.array([
        h_dist,
        v_dist,
    ])

def a_star_search(env, initial_obs):
    """ Perform A* search to decide whether to flap or not. """

    # Simulate both actions from the same state
    obs_0 = simulate_action(env, 0)

    obs_1 = simulate_action(env, 1)

    future_positions = {
        0: obs_0,  # Continue falling
        1: obs_1,  # Flap
    }
    costs = {}
    
    for action, future_obs in future_positions.items():
        g_cost = abs(initial_obs[0] - future_obs[0])**2 + abs(initial_obs[1] - future_obs[1])**2  # Using squared distance as cost
        h_cost = abs(future_obs[0])**2 + abs(future_obs[1])**2  # Heuristic: distance to the "goal"
        f_cost = g_cost + 10*h_cost  # Total cost
        costs[action] = f_cost   
    # Choose the action with the minimal cost
    best_action = min(costs, key=costs.get)
    return best_action

# Initialize environment
env = flappy_bird_gym.make("FlappyBird-v0")
obs = env.reset()

while True:
    # Decide next action with A* search
    action = a_star_search(env, obs)

    # Execute chosen action
    obs, reward, done, info = env.step(action)

    print("Plyer_vel {0}",env._game.player_vel_y)
    # Render the game
    env.render()
    time.sleep(1 / 30)  # Adjust FPS for human-friendly viewing

    # Break if the game is over
    if done:
        break

env.close()
