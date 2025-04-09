import pickle
import numpy as np
# Environment
import numpy as np
import random
import time
import pandas as pd
import pickle
import os
import json
from IPython.display import display, clear_output
import ipywidgets as widgets
import gymnasium as gym
from gymnasium import spaces
REWARD_GOAL = 100
REWARD_OBSTACLE = -100
REWARD_REVISIT = -10
REWARD_STEP = -0.5

with open("themes.json", "r", encoding="utf-8") as f:
    THEMES = json.load(f)
class MazeGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, maze_size, participant_id, enable_enemy, enable_trap_boost, max_steps):
        super().__init__()
        """
        initialize the maze_size, participant_id, enable_enemy, enable_trap_boost and max_steps variables
        """
        ###################################
        #      WRITE YOUR CODE BELOW      #
        # Assigning the configured values to the instance variables
        self.maze_size = maze_size
        self.participant_id = participant_id
        self.enable_enemy = enable_enemy # not used in main tasks 
        self.enable_trap_boost = enable_trap_boost
        self.max_steps = max_steps
        ###################################

        self.action_space = spaces.Discrete(4) # 4 actions: (0)up, (1)down, (2)left, (3)right
        self.observation_space = spaces.Tuple((
            spaces.Discrete(maze_size[0]),
            spaces.Discrete(maze_size[1])
        ))

        """
        generate  self.maze using the _generate_obstacles method
        make self.start as the top left cell of the maze and self.goal as the bottom right
        """
        ###################################
        #      WRITE YOUR CODE BELOW      #
        self.maze = self._generate_obstacles()
        self.start = (0, 0)
        self.goal = (self.maze_size[0]-1, self.maze_size[1]-1)
        self.visited = set() # keep track of visited cells within an episode
        ###################################

        if self.enable_trap_boost:
            self.trap_cells, self.boost_cells = self._generate_traps_and_boosts(self.maze)
        else:
            self.trap_cells, self.boost_cells = ([], [])

        self.enemy_cells = []
        self.current_step = 0
        self.agent_pos = None

        self.reset()

    def _generate_obstacles(self):
        """
        generates the maze with random obstacles based on the SR.No.
        """
        np.random.seed(self.participant_id)
        maze = np.zeros(self.maze_size, dtype=int)
        mask = np.ones(self.maze_size, dtype=bool)
        safe_cells = [
            (0, 0), (0, 1), (1, 0),
            (self.maze_size[0]-1, self.maze_size[1]-1), (self.maze_size[0]-2, self.maze_size[1]-1),
            (self.maze_size[0]-1, self.maze_size[1]-2)
        ]
        for row, col in safe_cells:
            mask[row, col] = False
        maze[mask] = np.random.choice([0, 1], size=mask.sum(), p=[0.9, 0.1])
        return maze

    def _generate_traps_and_boosts(self, maze):
        """
        generates special cells, traps and boosts. While training our agent,
        we want to pass thru more number of boost cells and avoid trap cells 
        """
        if not self.enable_trap_boost:
            return [], []
        exclusions = {self.start, self.goal}
        empty_cells = list(zip(*np.where(maze == 0)))
        valid_cells = [cell for cell in empty_cells if cell not in exclusions]
        num_traps = self.maze_size[0] * 2
        num_boosts = self.maze_size[0] * 2
        random.seed(self.participant_id)
        trap_cells = random.sample(valid_cells, num_traps)
        trap_cells_ = trap_cells
        remaining_cells = [cell for cell in valid_cells if cell not in trap_cells]
        boost_cells = random.sample(remaining_cells, num_boosts)
        boost_cells_ = boost_cells
        return trap_cells, boost_cells

    def move_enemy(self, enemy_pos):
        possible_moves = []
        for dx, dy in actions:
            new_pos = (enemy_pos[0] + dx, enemy_pos[1] + dy)
            if (0 <= new_pos[0] < self.maze_size[0] and
                0 <= new_pos[1] < self.maze_size[1] and
                self.maze[new_pos] != 1):
                possible_moves.append(new_pos)
        return random.choice(possible_moves) if possible_moves else enemy_pos

    def update_enemies(self):
        if self.enable_enemy:
            self.enemy_cells = [self.move_enemy(enemy) for enemy in self.enemy_cells]

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        empty_cells = list(zip(*np.where(self.maze == 0)))
        self.start = (0, 0)
        self.goal = (self.maze_size[0]-1, self.maze_size[1]-1)

        for pos in (self.start, self.goal):
            if pos in self.trap_cells:
                self.trap_cells.remove(pos)
            if pos in self.boost_cells:
                self.boost_cells.remove(pos)

        if self.enable_enemy:
            enemy_candidates = [cell for cell in empty_cells if cell not in {self.start, self.goal}]
            num_enemies = max(1, int((self.maze_size[0] * self.maze_size[1]) / 100))
            self.enemy_cells = random.sample(enemy_candidates, min(num_enemies, len(enemy_candidates)))
        else:
            self.enemy_cells = []

        self.current_step = 0
        self.agent_pos = self.start
        self.visited = set()


        return self.agent_pos, {}

    def get_reward(self, state):
        if state == self.goal:
            return REWARD_GOAL
        elif state in self.trap_cells:
            return REWARD_TRAP
        elif state in self.boost_cells:
            return REWARD_BOOST
        elif self.maze[state] == 1:
            return REWARD_OBSTACLE
        else:
            return REWARD_STEP

    def take_action(self, state, action):
        attempted_state = (state[0] + actions[action][0], state[1] + actions[action][1])
        if (0 <= attempted_state[0] < self.maze_size[0] and
            0 <= attempted_state[1] < self.maze_size[1] and
            self.maze[attempted_state] != 1):
            return attempted_state, False
        else:
            return state, True

    def step(self, action):
        self.current_step += 1
        next_state, wall_collision = self.take_action(self.agent_pos, action)
        if wall_collision:
            reward = REWARD_OBSTACLE
            next_state = self.agent_pos
        else:
            if self.enable_enemy:
                self.update_enemies()
            if self.enable_enemy and next_state in self.enemy_cells:
                reward = REWARD_ENEMY
                done = True
                truncated = True
                info = {'terminated_by': 'enemy'}
                self.agent_pos = next_state
                return self.agent_pos, reward, done, truncated, info
            else:
                revisit_penalty = REWARD_REVISIT if next_state in self.visited else 0
                self.visited.add(next_state)
                reward = self.get_reward(next_state) + revisit_penalty
        self.agent_pos = next_state

        if self.agent_pos == self.goal:
            done = True
            truncated = False
            info = {'completed_by': 'goal'}
        elif self.current_step >= self.max_steps:
            done = True
            truncated = True
            info = {'terminated_by': 'timeout'}
        else:
            done = False
            truncated = False
            info = {
                'current_step': self.current_step,
                'agent_position': self.agent_pos,
                'remaining_steps': self.max_steps - self.current_step
            }

        return self.agent_pos, reward, done, truncated, info

    def render(self, path=None, theme="racing"):
        icons = THEMES.get(theme, THEMES["racing"])
        clear_output(wait=True)
        grid = np.full(self.maze_size, icons["empty"])
        grid[self.maze == 1] = icons["obstacle"]
        for cell in self.trap_cells:
            grid[cell] = icons["trap"]
        for cell in self.boost_cells:
            grid[cell] = icons["boost"]
        grid[self.start] = icons["start"]
        grid[self.goal] = icons["goal"]
        if path is not None:
            for cell in path[1:-1]:
                if grid[cell] not in (icons["goal"], icons["obstacle"], icons["trap"], icons["boost"]):
                    grid[cell] = icons["path"]
        if self.agent_pos is not None:
            if grid[self.agent_pos] not in (icons["goal"], icons["obstacle"]):
                grid[self.agent_pos] = icons["agent"]
        if self.enable_enemy:
            for enemy in self.enemy_cells:
                grid[enemy] = icons["enemy"]
        df = pd.DataFrame(grid)
        print(df.to_string(index=False, header=False))

    def print_final_message(self, success, interrupted, caught, theme):
        msgs = THEMES.get(theme, THEMES["racing"]).get("final_messages", {})
        if interrupted:
            print(f"\n{msgs.get('Interrupted', '🛑 Interrupted.')}")
        elif caught:
            print(f"\n{msgs.get('Defeat', '🚓 Caught by enemy.')}")
        elif success:
            print(f"\n{msgs.get('Triumph', '🏁 Success.')}")
        else:
            print(f"\n{msgs.get('TimeOut', '⛽ Time Out.')}")
# Agent
class QLearningAgent:
    def __init__(self, maze_size, num_actions, alpha=0.1, gamma=0.99):
        """
        initialize self.num_actions, self.alpha, self.gamma
        initialize self.q_table based on number of states and number of actions
        """
        ###################################
        #      WRITE YOUR CODE BELOW      #
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros(maze_size + (num_actions,)) # Dimensions: (maze_height, maze_width, num_actions)
        ###################################
        

    def choose_action(self, env, state, epsilon):
        """
        returns an integer between [0,3]

        epsilon is a parameter between 0 and 1.
        It is the probability with which we choose an exploratory action (random action)
        Eg: ---
        If epsilon = 0.25, probability of choosing action from q_table = 0.75
                           probability of choosing random action = 0.25
        """
        ###################################
        #      WRITE YOUR CODE BELOW      #
        random_probability = random.uniform(0, 1)
        if random_probability < epsilon:
            action = random.randint(0, self.num_actions-1)
        # Otherwise, exploit: choose the best action based on current Q-values
        else:
            if not isinstance(state, tuple):
                pass
            # Find the action index with the highest Q-value for the current state
            action = np.argmax(self.q_table[state])
        return action
        ###################################


    def update(self, state, action, reward, next_state):
        """
        Use the Q-learning update equation to update the Q-Table
        """
        ###################################
        #      WRITE YOUR CODE BELOW      #
        current_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[next_state])
        # Q-learning update rule
        # new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

        # Update the Q-table with the newly calculated Q-value
        self.q_table[state][action] = new_q
        ###################################




with open('your_file.pkl', 'rb') as f:  # Replace with your actual file name
    data = pickle.load(f)
q_table = data['q_table']
participant_id = 23627
enable_trap_boost = False
enable_enemy = False
maze_size = (20, 20)
env = MazeGymEnv(maze_size, participant_id, enable_enemy, enable_trap_boost)
agent = QLearningAgent(maze_size, num_actions=4, alpha=0.7, gamma=0.9)  # Adjust alpha, gamma if different
agent.q_table = q_table  # Load the trained Q-table
REWARD_GOAL = 100
REWARD_OBSTACLE = -100
REWARD_REVISIT = -10
REWARD_STEP = -0.5
state = env.reset()  # Should return (0, 0)
for step in range(5):
    # Choose action greedily (epsilon = 0)
    action = np.argmax(agent.q_table[state])
    # Get current Q-value
    q_current = agent.q_table[state][action]
    # Take the step
    next_state, reward, done, _ = env.step(action)
    # Get max Q-value for next state
    max_q_next = np.max(agent.q_table[next_state])
    # Print values for the step
    print(f"Step {step + 1}:")
    print(f"  Current State: {state}")
    print(f"  Action: {action}")
    print(f"  Reward: {reward}")
    print(f"  Next State: {next_state}")
    print(f"  Q(s, a): {q_current}")
    print(f"  max Q(s', a'): {max_q_next}")
    # Update state for next iteration
    state = next_state
    if done:
        print("Goal reached early!")
        break