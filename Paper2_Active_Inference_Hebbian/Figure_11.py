import numpy as np
import matplotlib.pyplot as plt

# Define the environment class
class MazeEnvironment:
    def __init__(self, size=(5, 5), start=(0, 0), goal=(4, 4), obstacles=[]):
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.grid = np.zeros(size)
        self.grid[start] = 1  # Start position
        self.grid[goal] = 2  # Goal position
        for obstacle in obstacles:
            self.grid[obstacle] = -1  # Obstacles

    def display(self):
        plt.imshow(self.grid, cmap='binary')
        plt.title('Maze Environment')
        plt.colorbar()
        plt.show()

# Memory Encoding and Retrieval Class
class MemorySystem:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory_storage = []

    def memory_encoding(self, sensory_information):
        sensory_information = self._flatten_information(sensory_information)  # Convert to consistent format
        if not self.memory_storage:
            encoded_memory = sensory_information
        else:
            encoded_memory = self._apply_hebbian_learning_rule(sensory_information)
        self.memory_storage.append(encoded_memory)
        if len(self.memory_storage) > self.capacity:
            self.memory_storage.pop(0)

    def memory_retrieval(self, retrieval_cues):
        retrieval_cues = self._flatten_information(retrieval_cues)  # Convert to consistent format
        return self._apply_hopfield_network(retrieval_cues)

    def _flatten_information(self, information):
        # Flattening the information into a single numpy array
        return np.concatenate([np.ravel(np.array(info, dtype=np.float32)) for info in information])

    def _apply_hebbian_learning_rule(self, input_data):
        eta = 0.1
        encoded_memory = []
        input_data = np.array(input_data, dtype=np.float32)
        for other_data in self.memory_storage:
            other_data = np.array(other_data, dtype=np.float32)
            min_length = min(len(input_data), len(other_data))
            encoded_data = eta * input_data[:min_length] * other_data[:min_length]  # Ensure same length
            encoded_memory.append(encoded_data)
        return np.mean(encoded_memory, axis=0)  # Average encoding to maintain consistent shape

    def _apply_hopfield_network(self, retrieval_cues):
        retrieved_memory = []
        for cue in retrieval_cues:
            memory = self._retrieve_memory(cue)
            retrieved_memory.append(memory)
        return retrieved_memory

    def _retrieve_memory(self, cue):
        memory = [self._hopfield_network(cue, encoded_data) for encoded_data in self.memory_storage]
        return memory

    def _hopfield_network(self, cue, encoded_data):
        memory = self._sign_function(np.sum(cue * encoded_data))
        return memory

    def _sign_function(self, value):
        return 1 if value >= 0 else -1

    def memory_storage_info(self):
        print("Memory Storage Capacity:", self.capacity)
        print("Number of Encoded Memories:", len(self.memory_storage))

# Attentional Selection Class
class AttentionalSelection:
    def __init__(self):
        self.bottom_up_attention_weights = []
        self.top_down_attention_weights = []

    def bottom_up_attention(self, stimuli, weights):
        attention_scores = [stimulus * weight for stimulus, weight in zip(stimuli, weights)]
        return attention_scores

    def top_down_attention(self, attention_scores, priorities):
        modulated_scores = [score * priority for score, priority in zip(attention_scores, priorities)]
        return modulated_scores

# Distress Dynamics Class
class DistressDynamics:
    def __init__(self, alpha, beta, gamma, xi, epsilon, M):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.xi = xi
        self.epsilon = epsilon
        self.M = M

    def calculate_distress(self, D, C, S, W):
        distress_change = self.alpha * D + self.beta * C + self.gamma * S + self.xi * W
        markovian_term = self.epsilon * np.dot(self.M, D)
        return distress_change + markovian_term

    def update_distress(self, D, C, S, W, dt):
        for _ in range(10):
            distress_change = self.calculate_distress(D, C, S, W)
            D += distress_change * dt
            print("Distress level:", D)
        return D

# Define the combined agent class
class CombinedAgent:
    def __init__(self, environment, learning_rate=0.1, discount_factor=0.9):
        self.environment = environment
        self.position = environment.start
        self.q_values = np.zeros(environment.size + (4,))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.memory_system = MemorySystem(capacity=100)
        self.attentional_selection = AttentionalSelection()
        self.distress_dynamics = DistressDynamics(alpha=0.2, beta=0.3, gamma=0.1, xi=0.4, epsilon=0.5, M=np.array([[0.2, 0.3, 0.5], [0.4, 0.1, 0.5], [0.1, 0.2, 0.7]]))

    def choose_action(self):
        row, col = self.position
        possible_actions = ['up', 'down', 'left', 'right']
        if np.random.uniform(0, 1) < 0.1:  # Exploration
            return possible_actions[np.random.randint(len(possible_actions))]
        else:
            action_values = [self.q_values[row, col, self.get_action_index(action)] for action in possible_actions]
            return possible_actions[np.argmax(action_values)]

    def get_action_index(self, action):
        action_map = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        return action_map[action]

    def step(self):
        action = self.choose_action()
        next_position, reward = self.take_action(action)
        self.update_q_values(action, reward, next_position)
        self.memory_system.memory_encoding([self.position, self.get_action_index(action), reward, next_position])
        self.position = next_position
        return next_position, reward

    def take_action(self, action):
        row, col = self.position
        if action == 'up':
            new_row, new_col = row - 1, col
        elif action == 'down':
            new_row, new_col = row + 1, col
        elif action == 'left':
            new_row, new_col = row, col - 1
        elif action == 'right':
            new_row, new_col = row, col + 1

        # Check boundaries and obstacle positions
        if new_row < 0 or new_row >= self.environment.size[0] or new_col < 0 or new_col >= self.environment.size[1] or (new_row, new_col) in self.environment.obstacles:
            return (row, col), -1  # Invalid move, stay in place with penalty

        if (new_row, new_col) == self.environment.goal:
            return (new_row, new_col), 1  # Goal reached, reward = 1

        return (new_row, new_col), 0  # Valid move, no reward

    def update_q_values(self, action, reward, next_state):
        row, col = self.position
        next_row, next_col = next_state
        action_index = self.get_action_index(action)
        self.q_values[row, col, action_index] += self.learning_rate * (
            reward + self.discount_factor * np.max(self.q_values[next_row, next_col]) - self.q_values[row, col, action_index]
        )

    def navigate_maze(self, num_episodes=100):
        episode_rewards = []
        for episode in range(num_episodes):
            self.position = self.environment.start
            total_reward = 0
            while self.position != self.environment.goal:
                next_state, reward = self.step()
                total_reward += reward
            episode_rewards.append(total_reward)
        return episode_rewards

# Example usage
if __name__ == "__main__":
    # Define the maze environment
    maze_size = (5, 5)
    start_position = (0, 0)
    goal_position = (4, 4)
    obstacle_positions = [(1, 2), (2, 2), (3, 2)]
    maze = MazeEnvironment(size=maze_size, start=start_position, goal=goal_position, obstacles=obstacle_positions)

    # Display the maze environment
    maze.display()

    # Create and test the combined agent
    agent = CombinedAgent(environment=maze)
    episode_rewards = agent.navigate_maze(num_episodes=100)

    # Plot episode rewards
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Agent Performance')
    plt.show()
