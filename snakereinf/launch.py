from snakereinf.Snake import Snake
from SnakeEnv import SnakeEnv
from utils import perform_mc, show_games

# Winning everytime hyperparameters
grid_length = 3
n_episodes = 175000
epsilon = 0.9
gamma = 0.5
rewards = [-100000, -1, 10000, 100000]
# [Losing move, inefficient move, efficient move, winning move]

# Playing part
# game = Snake((800, 800), grid_length)
# game.start_interactive_game()

# Training part
env = SnakeEnv(grid_length=grid_length, with_rendering=False)
q_table = perform_mc(env, n_episodes, epsilon, gamma, rewards)


# Viz part
env = SnakeEnv(grid_length=grid_length, with_rendering=True)
show_games(env, 100, q_table)
