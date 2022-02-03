import numpy as np
import torch
from collections import deque
import math
import random

class Agent:

    # Function to initialise the agent
    def __init__(self):
        self.dqn = DQN()
        self.buffer = ReplayBuffer()

        # True when the replay buffer has been filled
        self.filledBuffer = False
        # Set the minibatch size
        self.minibatch_size = 850
        # Set the episode length
        self.episode_length = 150
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = [-999, -999]
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # Set the initial policy
        self.policy = 0.25 * np.ones((100, 100, 4))
        self.set_init_policy()
        # Discount factor used in Bellman equation
        self.gamma = 0.95
        # Initial epsilon value which allows for exploration
        self.epsilon = 0.6
        # Tracks the total number of episodes completed
        self.episodes = 0
        # Tracks the number of steps between each target network update
        self.update_t = 0
        # Used to fill the replay buffer
        self.reachGoal = False
        # Used to find out how many steps it took the greedy policy to reach goal
        self.steps = 0
        # True if the greedy policy reached the goal within 100 steps
        self.greedyGoal = False

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if( (self.num_steps_taken % self.episode_length == 0) and (self.num_steps_taken>0) ):
            if(self.filledBuffer):
                if not(self.greedyGoal):
                    self.episode_length = 150
                    self.episodes += 1
                    self.num_steps_taken = 0
                    self.steps = 0

                    # Increase the episode length to encourage more exploration
                    if(self.episodes > 30):
                        self.episode_length = 175

                    if(self.episodes > 45):
                        self.episode_length = 200
                    
                    # Check the greedy policy every 15 episodes
                    if((self.episodes % 15) == 0 and (self.episodes > 0)):
                        self.episode_length = 100

                    # Find the GLIE epsilon value
                    # dec_eps = 0 every 15 episodes to get the greedy policy
                    dec_eps = self.get_eps()
                    
                    # Update the policy when an episode ends
                    self.update_policy(dec_eps)
                return True

            else:   
                # Filling in the buffer
                if(self.reachGoal):
                    self.reachGoal = False
                    return True
                else:
                    return False
        else:
            return False

    # GLIE: Function to decrease epsilon as episode number increases
    def get_eps(self):
        if(self.episodes < 6):
            dec_eps = self.epsilon
        else:
            dec_eps = (self.epsilon) / ((self.episodes-4)**0.35)
        
        # Keeps epsilon high if the greedy policy fails to reach goal to encourage exploration
        if(self.episodes > 30):
            dec_eps = (self.epsilon) / ((self.episodes - 30)**0.35)

        if(self.episodes > 45):
            dec_eps = (self.epsilon) / ((self.episodes - 45)**0.35)

        if(self.episodes > 60):
            dec_eps = (self.epsilon) / ((self.episodes - 60)**0.35)
        
        if(self.episodes > 75):
            dec_eps = (self.epsilon) / ((self.episodes - 75)**0.35)
        
        if(self.episodes > 90):
            dec_eps = (self.epsilon) / ((self.episodes - 90)**0.35)

        # Greedy policy every 15 episodes
        if((self.episodes % 15) == 0 and (self.episodes > 0)):
            dec_eps = 0
        
        return dec_eps

    # Function to update the current policy every time an episode ends
    def update_policy(self, dec_eps):
        q_vals = self.dqn.get_q_values(self.buffer)
        max_actions = q_vals.argmax(axis=2)

        suboptimal_actions = dec_eps / 4

        for i in range(len(q_vals)):
            for j in range(len(q_vals[0])):
                new_policy = suboptimal_actions * np.ones(4)
                my_action = max_actions[i][j]
                new_policy[my_action] = (1 - dec_eps) + (dec_eps / 4)
                self.policy[i][j] = new_policy


    # Function to get the next action
    def get_next_action(self, state):

        action = self.choose_action()
        action = self.to_cont_action(action)

        self.num_steps_taken += 1
        self.state = state
        self.action = action
        return action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        self.steps += 1

        action = self.to_disc_action(self.action)
        # Convert the distance to a reward
        reward = self.get_reward(next_state, distance_to_goal, action)
        transition = (self.state, action, reward, next_state)
        self.buffer.append(transition)

        len_buffer = self.buffer.get_length()
        # Append weights index list
        self.buffer.weights_index.append(len_buffer-1)

        # Append weights of new transition as max of current weights
        if(len_buffer == 1):
            self.buffer.weights.append(2)
        else:
            if not(max(self.buffer.weights) > 1.99):
                max_weight = max(self.buffer.weights) + (0.02 * self.state[0])
            else:
                max_weight = max(self.buffer.weights)
            self.buffer.weights.append(max_weight)

        if(len_buffer < 15000):
            # Only filling in the replay buffer
            self.filledBuffer = False

            if(distance_to_goal < 0.02):
                self.reachGoal = True
                self.num_steps_taken = self.episode_length
        else:
            # Finished filling in the replay buffer. Start training
            self.filledBuffer = True

            # If not executing the greedy policy, train the network
            if (not(self.greedyGoal) and not((self.episodes % 15) == 0) ):
                # Weighted sampling
                samples, indexes = self.buffer.sampling(self.minibatch_size)
                # Add a small positive constant to the weights which is 5% of the largest weight
                self.dqn.w_epsilon = max(self.buffer.weights) * 0.05

                # Train the network
                delta = self.dqn.train_q_network(samples, self.gamma)

                # Update the transition weights based on delta
                for x,y in zip(indexes, delta):
                    self.buffer.weights[x] = y

                self.update_t += 1
                # Update the target network
                if(self.update_t == 10):
                    self.dqn.update_target()
                    self.update_t = 0
            
            # Executing the greedy policy
            if((self.episodes % 15) == 0):
                if( (math.isclose(next_state[0],self.state[0], rel_tol = 0.004)) and (math.isclose(next_state[1],self.state[1], rel_tol = 0.004)) and not(self.greedyGoal)):
                    self.num_steps_taken = self.episode_length
                    #print("Greedy STUCK; Dist: {}".format(distance_to_goal))
                    # Greedy policy is stuck
                
                if((distance_to_goal < 0.03) and (self.steps < 100) and not(self.greedyGoal)):
                    print("GREEDY Reached Goal in {} steps; Episode {}".format(self.steps, self.episodes))
                    self.greedyGoal = True
                    self.num_steps_taken = self.episode_length
                    # Stop training
    
    # Function to determine how much reward is associated to a transition
    def get_reward(self, sprime, dist, action):
        # Reward 1 based on distance to goal
        # Reward 2 based on x position (higher x position means higher rewards)
        # Total reward = (a * Reward 1) + (b * Reward 2)
        # Where a and b are weighting factors

        a = 0.35
        b = 0.65

        # Reward based on distance to goal
        factor1 = -1.5
        power1 = -1.5
        reward1 = factor1/((dist)**power1)

        # Reward based on x position
        factor2 = -2
        reward2 = factor2 * (1 - self.state[0])

        total_reward = (a * reward1) + (b * reward2)
        
        # If the agent is very close to the goal, add more rewards
        if(dist < 0.1):
            total_reward = total_reward + 0.4

        if(dist < 0.05):
            total_reward = total_reward + 0.6

        # If agent moves West, that's not good
        if (action == 3):
            total_reward = total_reward - 0.2
        
        # If agent stays in the same state, that's not good
        if( (math.isclose(sprime[0],self.state[0], rel_tol = 0.004)) and (math.isclose(sprime[1],self.state[1], rel_tol = 0.004)) ):
            total_reward = total_reward - 0.03

        return total_reward
    
    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        
        row, col = self.state_to_rowcol(state)
        
        policy = self.policy[round(col)][round(row)]
        
        action = np.argmax(policy)
        #print("({},{}); A: {}".format(round(col),round(row),action))
        action = self.to_cont_action(action)

        return action
    
    # Defining the initial policy
    def set_init_policy(self):
        init_pol = [0.275, 0.3, 0.275, 0.15]

        for i in range(len(self.policy)):
            for j in range(len(self.policy[0])):
                self.policy[i][j] = init_pol

    # Choose an action based on the current policy
    def choose_action(self):
        if( (self.state[0]==-999) and (self.state[1]==-999) ):
            action = 1
        else:
            row, col = self.state_to_rowcol(self.state)
            policy = self.policy[round(col)][round(row)]

            action = random.choices(np.arange(4), weights = policy)[0]
        return action

    # Mapping the discrete actions to continuous actions
    def to_cont_action(self, action):
        if action == 0:
            cont_action = np.array([0, 0.02], dtype=np.float32)
        elif action == 1:
            cont_action = np.array([0.02, 0], dtype=np.float32)
        elif action == 2:
            cont_action = np.array([0, -0.02], dtype=np.float32)
        elif action == 3:
            cont_action = np.array([-0.02, 0], dtype=np.float32)
        else:
            print("Invalid action")
            cont_action = np.array([0, 0], dtype=np.float32)
        
        return cont_action

    # Mapping the continuous actions to discrete actions
    def to_disc_action(self, action):
        if( (action[0]==0) and (action[1]>0) ):
            disc_action = 0
        elif( (action[0]>0) and (action[1]==0) ):
            disc_action = 1
        elif( (action[0]==0) and (action[1]<0) ):
            disc_action = 2
        elif( (action[0]<0) and (action[1]==0) ):
            disc_action = 3
        else:
            print("Invalid action")
            disc_action = 4
        return disc_action

    # Mapping the states to columns and rows of the Q function for easy indexing
    def state_to_rowcol(self, state):
        col = math.floor(state[0] * 100)
        row = math.floor(state[1] * 100)
        return row, col


class Network(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension):
        super(Network, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=90)
        self.layer_2 = torch.nn.Linear(in_features=90, out_features=125)
        self.layer_3 = torch.nn.Linear(in_features=125, out_features=90)
        self.output_layer = torch.nn.Linear(in_features=90, out_features=output_dimension)

    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_3_output)
        return output


class DQN:

    def __init__(self):
        self.q_network = Network(input_dimension=2, output_dimension=4)
        self.t_network = Network(input_dimension=2, output_dimension=4)
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.002)
        self.t_network.load_state_dict(self.q_network.state_dict())
        self.w_epsilon = 0.05

    def update_target(self):
        self.t_network.load_state_dict(self.q_network.state_dict())

    def train_q_network(self, transition, gamma):
        self.optimiser.zero_grad()
        loss, delta = self._calculate_loss(transition, gamma)
        loss.backward()
        self.optimiser.step()
        return delta

    def _calculate_loss(self, transition, gamma):
        # Double Q Deep Learning
        s_tensor, a_tensor, r_tensor, sprime_tensor = zip(*transition)

        s_tensor = torch.tensor(s_tensor, dtype=torch.float32)
        a_tensor = torch.tensor(a_tensor)
        r_tensor = torch.tensor(r_tensor, dtype=torch.float32)
        sprime_tensor = torch.tensor(sprime_tensor, dtype=torch.float32)

        # Get Q value from Q network indexed by action taken
        s_prediction = self.q_network.forward(s_tensor).gather(dim=1, index=a_tensor.unsqueeze(-1)).squeeze(-1)

        # Get max actions from target network Q s_prime predictions
        with torch.no_grad():
            max_actions = (self.t_network.forward(sprime_tensor)).argmax(1)

        # Get Q value using max action from target network
        sprime_prediction = self.q_network.forward(sprime_tensor).gather(dim=1, index=max_actions.unsqueeze(-1)).squeeze(-1)
        sprime_prediction = r_tensor + (gamma * sprime_prediction)

        # New weights
        delta = abs(sprime_prediction - s_prediction) + self.w_epsilon
        delta = delta.detach().numpy()

        # Calculate loss
        loss = torch.nn.MSELoss()(s_prediction, sprime_prediction)
        return loss, delta

    def get_q_values(self, buffer):

        q_vals = np.zeros((100, 100, 4))
        init_q = [-50, -40, -50, -60]

        for i in range(len(q_vals)):
            for j in range(len(q_vals[0])):
                q_vals[i][j] = init_q
        
        samples = buffer.get_all()
        state_tensor = [i[0] for i in samples]
        state_tensor = np.unique(state_tensor, axis=0)

        for i in state_tensor:
            col = math.floor(i[0] * 100)
            row = math.floor(i[1] * 100)

            state = torch.tensor(i)
            q_vals[round(col)][round(row)] = self.q_network.forward(state).detach().numpy()
        
        return q_vals


class ReplayBuffer:

    def __init__(self):
        self.buffer = deque(maxlen=75000)
        self.weights = []
        self.weights_index = []

    def append(self, input_tuple):
        self.buffer.append(input_tuple)

    def sampling(self, minibatch_size):
        if(minibatch_size > len(self.buffer)):
            print("Buffer too short")
        else:
            # Weighted sampling
            indexes = random.choices(self.weights_index, self.weights, k=minibatch_size)
            samples = [self.buffer[i] for i in indexes]

            return samples, indexes
    
    def get_length(self):
        return len(self.buffer)

    def get_all(self):
        return self.buffer