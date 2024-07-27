import numpy as np
import matplotlib.pyplot as plt

class MazeEnvironment:
    def __init__(self):
        self.grid = np.zeros((5, 5))
        self.grid[0, 0] = 0.75  # Gray square in top left
        self.grid[4, 4] = 2.0   # Black square in bottom right
        self.grid[1:4, 2] = -1.0  # White squares in the middle

    def render(self):
        plt.imshow(self.grid, cmap='gray', vmin=-1, vmax=2)
        plt.colorbar()
        plt.title("Maze Environment")
        plt.show()

class Agent:
    def __init__(self, env):
        self.env = env
        self.position = (0, 0)
        self.total_reward = 0
        self.history = []

    def step(self, action):
        row, col = self.position
        if action == 'up' and row > 0:
            row -= 1
        elif action == 'down' and row < 4:
            row += 1
        elif action == 'left' and col > 0:
            col -= 1
        elif action == 'right' and col < 4:
            col += 1
        
        self.position = (row, col)
        reward = self.env.grid[row, col]
        self.total_reward += reward
        self.history.append((self.position, reward))
        return reward

    def render(self):
        env_copy = self.env.grid.copy()
        row, col = self.position
        env_copy[row, col] = 1.5  # Indicate agent's position
        plt.imshow(env_copy, cmap='gray', vmin=-1, vmax=2)
        plt.colorbar()
        plt.title("Maze Environment with Agent")
        plt.show()

# Initialize environment and agent
env = MazeEnvironment()
agent = Agent(env)

# Render the environment
env.render()
agent.render()

# Sample agent performance
performance = []
for episode in range(100):
    agent.position = (0, 0)
    agent.total_reward = 0
    for _ in range(10):  # Assume 10 steps per episode
        action = np.random.choice(['up', 'down', 'left', 'right'])
        agent.step(action)
    performance.append(agent.total_reward)

# Plot performance
plt.plot(performance)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Agent Performance')
plt.show()
