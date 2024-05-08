import socket
import torch
from torch import nn
import torch.nn.functional as F
import gym
import numpy as np
from copy import deepcopy
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
from IPython import display
from gym.wrappers import TimeLimit
from torch.nn import BatchNorm1d

######
import csv
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


start_from_check_point = False
start_from_ep = 54

no_balls = 4

state_dim = 4
act_dim = 3

#Limits the number of time steps per episode to avoid hovering 
step_limit=150

#Action in pool is: angle between 0 and 360, strength between 0 and 1
action_limit = [13, 7, 1]

gamma = 0.99

#This helps with exploration vs eploitation
"""alpha_max = 0.4
alpha_min=0.01
alpha=alpha_max

learning_rate_max = 0.001
learning_rate_min=0.0001
learning_rate=learning_rate_max"""
alpha=0.1
# learning_rate=0.0001

actor_lr = 0.0002
critic_lr = 0.0003
#Used for soft update of the target critics
tau = 0.001

epoch = 1000
time_steps = epoch * 6
test_episodes = 1
initial_steps = 50
buffer_size = 1000000
batch_size = 256
noise_std = 0.1

if start_from_check_point:
    initial_steps = 0
    
def normalize_state(state):
    print(state)
    min_val = -20
    max_val = 20
    state[0] = (state[0] - min_val) / (max_val - state[0])
    state[1] = (state[1] - min_val) / (max_val - state[1])
    state[2] = (state[2] - min_val) / (max_val - state[2])
    state[3] = (state[3] - min_val) / (max_val - state[3])
    
    return state

def check_wait(s):
    waiting = True
    while waiting: 
        data = s.recv(1024).decode()
        if data.startswith("WAIT"):
            waiting = True
        else:
            waiting = False 
            #print("stop waiting")

def receive_state(s):
    check_wait(s)
    data = s.recv(1024).decode()
    parts = data.split(',')
    state = [float(x) for x in parts[:-2]]
    state = normalize_state(state)
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

def send_instruction(s, instruction):
    #print("sending instruction")
    check_wait(s)
    #s.sendall(f"{instruction[0]},{instruction[1]}".encode())
    s.sendall(f"INSTRUCTION,{instruction[0]},{instruction[1]},{instruction[2]}".encode())
    #print(f"Sent instruction: {instruction}")
    
def reset_env(s):
    check_wait(s)
    s.sendall("RESET".encode())
    #print("Sent Reset command")


# Create a directory for TensorBoard data
# Setting up logging tools
writer = SummaryWriter('runs/pool_experiment_' + datetime.now().strftime("%Y%m%d-%H%M%S"))
csv_file = open('training_metrics.csv', 'a', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Episode', 'TotalReward'])

"""def save_checkpoint(actor, critic1, critic2, actor_optimizer, critic1_optimizer, critic2_optimizer, filename):
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic1_state_dict': critic1.state_dict(),
        'critic2_state_dict': critic2.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'critic1_optimizer_state_dict': critic1_optimizer.state_dict(),
        'critic2_optimizer_state_dict': critic2_optimizer.state_dict(),
    }, 'models.pth')"""


import os

def save_checkpoint(actor, critic, actor_optimizer, critic_optimizer, episode):
    save_path = 'checkpoints'  # Directory where you want to save your checkpoints
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists

    # Create a unique filename for each checkpoint
    filename = f'checkpoint_episode_{episode}.pth'
    filepath = os.path.join(save_path, filename)  # Full path to the file

    # Save the checkpoint
    torch.save({
        'episode': episode,
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': critic_optimizer.state_dict()
    }, filepath)

    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(actor, critic, actor_optimizer, critic_optimizer, filename):
    # Load the checkpoint
    checkpoint = torch.load(filename)

    episode = checkpoint['episode']

    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

    print(f"Checkpoint loaded from {filename}")
    return episode
#######


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using {}".format(device))

#If you see Using cuda then gpu is used
## Global variables

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Actor, self).__init__()
        
        #Build the actor neural network
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.bn1 = BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = BatchNorm1d(hidden_size)
        self.mean_fc = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        #Use tanh for mean so it is between -1 and 1
        
       # action = torch.tanh(self.mean_fc(x))
       
        action_sigmoid = torch.tanh(0.1 * self.mean_fc(x))
       
        
        #Make sure the action is in the action limit by transforming it with tanh(which is between -1 and 1)
        #and multiplying with the action_limit
        #!!! This assumes that the action is in between -action_limit and action_limit
        
        # action_sigmoid = torch.sigmoid(action)
        
        # action_sigmoid[0] = action_limit[0] * action_sigmoid[0]
        # action_sigmoid[1] = action_limit[1] * action_sigmoid[1]

        return action_sigmoid
    # , log_prob
    
    def clip_gradients(self, clip_value):
        
        # Added this function because the gradients would explode and make the actor network return NaN
        # This is used to clip the gradients between given values
        
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-clip_value, clip_value)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Critic, self).__init__()
        
        #Build the critic neural network
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.bn1 = BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #Make the autput the right shape
        return torch.squeeze(x, -1)                

class ReplayBuffer:
    def __init__(self, state_dim, act_dim, size):
        self.state_buf = np.zeros((size, state_dim), dtype=np.float32)
        self.next_state_buf = np.zeros((size, state_dim), dtype=np.float32)
        self.action_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.reward_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, state, action, reward, next_state, done):
        self.state_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = next_state
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
    


    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(state=self.state_buf[idxs],
                     next_state=self.next_state_buf[idxs],
                     action=self.action_buf[idxs],
                     reward=self.reward_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

def DDPG():
    global step_counter
    #global alpha, learning_rate
    #Define replay buffer
    replay_buffer = ReplayBuffer(state_dim=state_dim, act_dim=act_dim, size=buffer_size)
    
    #Define actor and critics
    actor = Actor(state_dim, act_dim)#
    critic = Critic(state_dim, act_dim)#
    # critic2 = Critic(state_dim, act_dim)#
    #.to(device)
    #Define targets (for SAC only the critics have targets)
    target_critic = deepcopy(critic)#
    target_actor = deepcopy(actor)
    # target_critic2 = deepcopy(critic2)#
    
    #Optimizers
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    # critic2_optimizer = torch.optim.Adam(critic2.parameters(), lr=learning_rate)
    
    def get_action(state):
        # action= actor(torch.as_tensor(state, dtype=torch.float32))
        # action = action.detach().numpy()
        # action[0] = action_limit[0] * action[0]
        # action[1] = action_limit[1] * action[1]
        # return action
        action = actor(torch.as_tensor(state, dtype=torch.float32))
    
        # Add Gaussian noise to the mean action
        noise = torch.randn_like(action) * noise_std
        noisy_action = torch.clamp(action + noise, -1, 1)  # Ensure actions are within [-1, 1]
        print("hi")
        #print(action)

        

        # Transform action[0] to be between 0 and 360
        # noisy_action[0] = (noisy_action[0] + 1) * 180
        
        # Transform action[1] to be between 0 and 1
        # noisy_action[1] = (noisy_action[1] + 1) / 2
        noisy_action = noisy_action.detach().numpy()
        noisy_action[0] = action_limit[0] * noisy_action[0]
        noisy_action[1] = action_limit[1] * noisy_action[1]
        noisy_action[2] = 1/(1 + np.exp(-noisy_action[2]))

        
        return noisy_action
    
    def get_action_test(state):
        # action= actor(torch.as_tensor(state, dtype=torch.float32))
        # action = action.detach().numpy()
        # action[0] = action_limit[0] * action[0]
        # action[1] = action_limit[1] * action[1]
        # return action
        action = actor(torch.as_tensor(state, dtype=torch.float32))
    
        # Add Gaussian noise to the mean action
        noise = torch.randn_like(action) * noise_std
        #noisy_action = torch.clamp(action + noise, -1, 1)  # Ensure actions are within [-1, 1]
        print("hi")
        #print(action)
        noisy_action = action
        

        # Transform action[0] to be between 0 and 360
        # noisy_action[0] = (noisy_action[0] + 1) * 180
        
        # Transform action[1] to be between 0 and 1
        # noisy_action[1] = (noisy_action[1] + 1) / 2
        noisy_action = noisy_action.detach().numpy()
        noisy_action[0] = action_limit[0] * noisy_action[0]
        noisy_action[1] = action_limit[1] * noisy_action[1]
        noisy_action[2] = 1/(1 + np.exp(-noisy_action[2]))

        
        return noisy_action
    
    #This function is used to update the actor and the critics
    def step(state, action, reward, next_state, done):
        torch.autograd.set_detect_anomaly(True)

        target_action = target_actor(next_state)
        target_Q = target_critic(next_state, target_action)
        target_Q = reward + (1 - done) * gamma * target_Q.detach()

        current_Q = critic(state, action)
        
        cricic_loss = F.mse_loss(current_Q, target_Q)
        
        #Update the first critic using its loss and gradient descent
        critic_optimizer.zero_grad()
        cricic_loss.backward()
        critic_optimizer.step()
        
        for p in critic.parameters():
            p.requires_grad = False
            
            
        #Update the actor using its loss and gradient descent

        actor_optimizer.zero_grad()
        acotr_loss = -critic(state, actor(state)).mean()
        acotr_loss.backward()
        actor_optimizer.step()
        
        # actor.clip_gradients(1)
            
        for p in critic.parameters():
            p.requires_grad = True
        for p in actor.parameters():
            p.requires_grad = True
        
        #Update the target critics by taking their weights, multiplying by tau, and then adding (1-tau) * the weights of the respective critic 
        #Soft update by using tau
        with torch.no_grad():
            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    total_test_reward = []

    def send_action(s, action):
        send_instruction(s, action)
        next_state, reward, done = receive_state(s)
        return next_state, reward, done
    
    def reset(s):
        reset_env(s)
        state, _, _ = receive_state(s)
        return state

    def test_agent(s):
        mean_reward = []
        for j in range(test_episodes):
            print(j)
            state, done = reset(s), False
            ep_len = 0
            tot_reward = 0
            while not(done or (ep_len == step_limit)):
                action = get_action_test(state)
                state, reward, done = send_action(s, action)
                tot_reward += reward
                ep_len += 1
            mean_reward.append(tot_reward)
        total_test_reward.append(np.sum(mean_reward)/test_episodes)
    
    def sample_action():
        action = []
        for i in range(act_dim-1):
            action.append(np.random.uniform(-action_limit[i], action_limit[i]))
        action.append(np.random.uniform(-0, action_limit[act_dim-1]))
        return np.asarray(action)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        total_rewards = []
        t_rew = 0
        step_counter=0
        episode = 0
        host = '127.0.0.1'
        port = 8888
        s.connect((host, port))
        print("Connected to the server.")
        state = reset(s)

        if start_from_check_point:
            filename = f"checkpoints/checkpoint_episode_{start_from_ep}.pth"  # Specify the path to your checkpoint file
            episode = load_checkpoint(actor, critic, actor_optimizer, critic_optimizer, filename)


        for i in range(1,time_steps):
            if i%epoch == 0:
                test_agent(s)
                state = reset(s)

            if i < initial_steps:
                #We want to have 'initial_steps' samples in the replay buffer before we start training, so we use random ones
                action = sample_action()
                # print("Action=", action)
            else:
                    # if i%epoch == 0:
                    #     print("testing...")
                    #     test_agent(s)
                    #     state = reset(s)
                    # else:
                action = get_action(state)
                #print(action)
            # print("Action=", action)
            #Take a step using the action
            
            print(i)

            next_state, reward, done = send_action(s, action)
            
            step_counter+=1
            if step_counter >= step_limit:
                done = True
                reward=reward-50
                
            # print(i)

            replay_buffer.store(state, action, reward, next_state, done)

            #Adding to keep track of total reward
            t_rew += reward
            
            #Update the state
            state = next_state
            
            if done:
                #When done add total reward to a list and set to 0 again, then reset env
                total_rewards.append(t_rew)
                print("=====================================================")
                print(f"Episode {episode} finished with reward: {t_rew}")
                print(reward)
                #print("info", info['env'].lander_position[0])
                writer.add_scalar('Total Reward', t_rew, episode)
                csv_writer.writerow([episode + 1, t_rew])
                csv_file.flush()
                print("step_counter=", step_counter)
                print("curreny time steps=", i)
                #print("State=", state)
                step_counter=0

        # Reset the environment and reward tracker
                t_rew = 0
                episode+=1
                if episode % 50 == 0:  # Save every 50 episodes
                    save_checkpoint(actor, critic, actor_optimizer, critic_optimizer, episode)
                state = reset(s)

                state = torch.as_tensor(state, dtype=torch.float32)#
            if i >= initial_steps:
                #If we are training, then take a minibatch and call function step
                batch = replay_buffer.sample_batch(batch_size)
                #batch = {k: v.to(device) for k, v in batch.items()} #
                step(batch['state'], batch['action'], batch['reward'], batch['next_state'], batch['done'])

        # finish training
        save_checkpoint(actor, critic, actor_optimizer, critic_optimizer, episode)
    csv_file.close()
    writer.close()
    print(total_rewards)

    # print("testing...")
    # test_agent(s)
    
    """
    plt.plot(total_rewards)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f'plot_{timestamp}.png'
"""
    plt.figure()  # Create a new figure
    plt.plot(total_rewards, label='Training Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Total Training Rewards Over Time')
    plt.legend()
    plt.grid()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f'training_rewards_{timestamp}.png'
    plt.savefig(filename)
    plt.show()
    plt.savefig(filename)

    # Plot test rewards
    plt.figure()  # Create a new figure
    plt.plot(total_test_reward, label='Test Episode Rewards')
    plt.xlabel('Test Episode')
    plt.ylabel('Reward')
    plt.title('Total Test Rewards Over Time')
    plt.legend()
    plt.grid()
    filename = f'test_rewards_{timestamp}.png'

    plt.show()

    plt.savefig(filename)
    plt.show()

"""state,_ = env.reset()
env.render()
for j in range(time_steps):
    action, _ = actor(torch.as_tensor(state, dtype=torch.float32))
    action = action.detach().numpy()

    next_state, reward, done,_,_ = env.step(action)
    display.clear_output(wait=True)
    plt.imshow(env.render())
    display.display(plt.gcf())
    if done:
        state,_ = env.reset()"""
    


DDPG() 