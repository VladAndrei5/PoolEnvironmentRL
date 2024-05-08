import socket
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value
    

class DDPG:
    
    def check_wait(self, s):
        waiting = True
        while waiting: 
            data = s.recv(1024).decode()
            if data.startswith("WAIT"):
                waiting = True
            else:
                waiting = False 
                #print("stop waiting")

    def receive_state(self, s):
        self.check_wait(s)
        data = s.recv(1024).decode()
        parts = data.split(',')
        state = [float(x) for x in parts[:-2]]
        reward = float(parts[-2])
        terminal = (parts[-1])
        if terminal == "False":
            terminal = False
        else:
            terminal = True
        # for some reason the terminal is reversed
        s.sendall("RECIEVEDSTATE".encode())    
        #print(f"Received response: state={state}, reward={reward}, terminal={terminal}")
        return state, reward, terminal        

    def send_instruction(self, s, instruction):
        #print("sending instruction")
        self.check_wait(s)
        #print(instruction)
        #s.sendall(f"{instruction[0]},{instruction[1]}".encode())
        s.sendall(f"INSTRUCTION,{instruction[0]},{instruction[1]},{instruction[2]}".encode())
        #print(f"Sent instruction: {instruction}")
        
    def close_server(self, s):
        self.check_wait(s)
        for i in range(200):
            s.sendall("DISCONNECT".encode())
        
    def reset_env(self, s):
        self.check_wait(s)
        s.sendall("RESET".encode())
        #print("Sent Reset command")
        
    def send_action(self, s, action):
        self.send_instruction(s, action)
        next_state, reward, done = self.receive_state(s)
        return next_state, reward, done
    
    def reset(self, s):
        self.reset_env(s)
        state, _, _ = self.receive_state(s)
        return state  
    
    

    
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, gamma, tau, alpha):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        self.update_target_networks(1.0)
        
    def select_action_test(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy().squeeze(0)
        
        action[0] = action_limit[0] * action[0]
        action[1] = action_limit[1] * action[1]
        action[2] = 1/(1 + np.exp(-action[2]))
        return action
        
    def select_action(self, state, episode):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy().squeeze(0)
        
        noise_scale = max(0.1, 1.0 - episode / num_episodes)
        
        exploration_noise = np.random.normal(loc=0, scale=noise_scale, size=action.shape)
        
        
        action = action + exploration_noise
        action = np.clip(action, -1, 1)
        
        
        action[0] = action_limit[0] * action[0]
        action[1] = action_limit[1] * action[1]
        action[2] = 1/(1 + np.exp(-action[2]))
        return action
    
    def update_target_networks(self, tau):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def train_step(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = torch.FloatTensor(action).unsqueeze(0)
        reward = torch.FloatTensor([reward])
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        done = torch.FloatTensor([done])
        
        # Compute the target Q-value
        with torch.no_grad():
            next_action = self.target_actor(next_state)
            target_q = self.target_critic(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # Compute the current Q-value
        current_q = self.critic(state, action)
        
        # Compute the critic loss
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Compute the actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update the target networks using soft update
        self.update_target_networks(self.alpha)

    def train(self, num_episodes, max_steps, test_interval, num_test_episodes):
        episode_rewards = []
        test_rewards = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            host = '127.0.0.1'
            port = 8888
            s.connect((host, port))
            print("Connected to the server.")

            for episode in range(num_episodes):
                state = self.reset(s)
                episode_reward = 0
                
                for step in range(max_steps):
                    action = self.select_action(state, episode)
                    #print(action)
                    next_state, reward, done = self.send_action(s, action)
                    
                    self.train_step(state, action, reward, next_state, done)
                    
                    state = next_state
                    episode_reward += reward
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
                print(f"Episode train: {episode+1}, Reward: {episode_reward}")
                
                if (episode + 1) % test_interval == 0:
                    test_reward = self.test(s, num_test_episodes, max_steps)
                    test_rewards.append(test_reward)
                    
            self.plot_results(episode_rewards, test_rewards, test_interval, num_test_episodes)        
            self.close_server(s)
            

            
            

    def test(self, s, num_episodes, max_steps):
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.reset(s)
            episode_reward = 0
            
            for step in range(max_steps):
                action = self.select_action_test(state)
                next_state, reward, done = self.send_action(s, action)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
        
        avg_reward = np.mean(episode_rewards)
        print(f"Test Episodes: {num_episodes}, Avg. Reward: {avg_reward}")
        return avg_reward, episode_rewards
        
    def plot_results(self, episode_rewards, test_rewards, test_interval, num_test_episodes):
        plt.figure(figsize=(12, 8))
        
        # Plot training rewards
        plt.subplot(2, 1, 1)
        plt.plot(range(1, len(episode_rewards) + 1), episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards')
        
        # Plot testing rewards
        plt.subplot(2, 1, 2)
        test_episodes = [test_interval * (i + 1) for i in range(len(test_rewards))]
        
        for i in range(len(test_rewards)):
            rewards = test_rewards[i][1]  # Get the list of episode rewards for each test
            if len(rewards) > 0:
                plt.plot(range(1, len(rewards) + 1), rewards, label=f'Test {i+1}')
        
        plt.xlabel('Test Episode')
        plt.ylabel('Reward')
        plt.title('Testing Rewards')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('ddpg_results.png')
        plt.show()
            
            
no_balls = 1
state_per_ball = 2
additional_states = 0


state_dim = no_balls * state_per_ball + additional_states  # Dimension of the state space
action_dim = 3  # Dimension of the action space
actor_lr = 1e-4  # Learning rate for the actor network
critic_lr = 1e-3  # Learning rate for the critic network
gamma = 0.99  # Discount factor
tau = 0.005  # Soft update factor for target networks
alpha = 0.001  # Added alpha parameter for soft update
num_episodes = 200  # Number of training episodes
max_steps = 1000  # Maximum number of steps per episode
test_interval = 50  # Number of episodes between each test
num_test_episodes = 2  # Number of episodes to run during each test

action_limit = [11, 6, 1]

# Create an instance of the DDPG agent
agent = DDPG(state_dim, action_dim, actor_lr, critic_lr, gamma, tau, alpha)

# Start training
agent.train(num_episodes, max_steps, test_interval, num_test_episodes)