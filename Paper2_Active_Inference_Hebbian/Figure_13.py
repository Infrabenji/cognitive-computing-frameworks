class AgentWithRE(Agent):
    def __init__(self, env):
        super().__init__(env)

    def compute_decision_function(self, sense_data, actions):
        best_action = None
        max_fitness = float('-inf')
        for action in actions:
            expected_fitness = 0
            for s in range(5):
                for s_prime in range(5):
                    prob_s = 1/5  # Simplified probability
                    prob_d_given_s = 1/5
                    prob_s_prime_given_s_and_a = 1/5
                    v_s_s_prime = self.env.grid[s_prime, s_prime]
                    expected_fitness += prob_s * prob_d_given_s * prob_s_prime_given_s_and_a * v_s_s_prime
            if expected_fitness > max_fitness:
                max_fitness = expected_fitness
                best_action = action
        return best_action

    def step_with_re(self):
        actions = ['up', 'down', 'left', 'right']
        sense_data = self.position
        action = self.compute_decision_function(sense_data, actions)
        return self.step(action)

# Initialize environment and agent with RE
agent_re = AgentWithRE(env)

# Sample agent performance with RE
performance_re = []
for episode in range(100):
    agent_re.position = (0, 0)
    agent_re.total_reward = 0
    for _ in range(10):  # Assume 10 steps per episode
        agent_re.step_with_re()
    performance_re.append(agent_re.total_reward)

# Plot performance with RE
plt.plot(performance_re)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Agent Performance with RE')
plt.show()
