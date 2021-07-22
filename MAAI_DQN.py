import      glob
import      io
import      base64
import      gym
import      ma_gym
import      threading
import      random
import      torch
import      torch.nn                as      nn
import      torch.optim             as      optim
import      torch.nn.functional     as      F
import      torchvision.transforms  as      T
import      matplotlib.pyplot       as      plt
import      numpy                   as      np
from        ma_gym.wrappers         import  Monitor
from        IPython.display         import  HTML
from        IPython                 import  display             as  ipythondisplay
from        pyvirtualdisplay        import  Display
from        tqdm                    import  tqdm
from        collections             import  namedtuple, deque

# display = Display(visible=0, size=(1400, 900))
# display.start()

"""
Utility functions to enable video recording of gym environment and displaying it
To enable video, just do "env = wrap_env(env)"
"""

class QNetwork(nn.Module):
    def __init__(self, observation_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1         = nn.Linear(observation_size, 64)
        self.fc2         = nn.Linear(64, 64)
        self.fc3         = nn.Linear(64, 64)
        self.q_value_out = nn.Linear(64, action_size)

    def forward(self, x):
        x       = F.relu(self.fc1(x))
        x       = F.relu(self.fc2(x))
        x       = F.relu(self.fc3(x))
        Q_value = self.q_value_out(x)
        return Q_value # Action as index

class DQNAgent:
    def __init__(self, agent_id, observation_size=3, action_size=5):
        self.agent_id           = agent_id
        self.action_size        = action_size
        self.train_step         = 0
        self.LR                 = 1e-3
        self.tau                = 0.2
        self.gamma              = 0.95
        self.batch_size         = 64
        self.buffer_size        = 10000
        self.UPDATE_EVERY       = 4
        self.agent_t_step       = 0 # Agent time step

        self.replay_memory      = ReplayBuffer(self.buffer_size, self.batch_size)           # buffer_size: size of memory --> number of experience, 
                                                                                            # batch_size:  number of experience from the memory used for learning

        self.Q_network          = QNetwork(observation_size, action_size)                   # create the network, input observation
        self.Q_target_network   = QNetwork(observation_size, action_size)                   # build up the target network
        self.Q_target_network.load_state_dict(self.Q_network.state_dict())                  # load the weights into the target networks
        self.Q_optim            = torch.optim.RMSprop(self.Q_network.parameters(), lr=self.LR) # create the optimizer

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.Q_target_network.parameters(), self.Q_network.parameters()):
            target_param.data.copy_(self.tau*param.data + (1 - self.tau)*target_param.data)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.replay_memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.agent_t_step = (self.agent_t_step + 1) % self.UPDATE_EVERY
        if self.agent_t_step == 0:
            # If enough samples are available in memory, get radom subset and learn
            if len(self.replay_memory) > self.batch_size:
                experience = self.replay_memory.sample()
                self.learn(experience, self.gamma)
    
    def select_action(self, o, epsilon):
        if np.random.uniform() < epsilon: # Exploration
            u      = np.int64(np.random.randint(0,4))
        else: # Exploitation
            self.Q_network.eval()
            with torch.no_grad():
                inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
                u      = np.argmax(self.Q_network.forward(inputs)[0].detach().numpy())
            self.Q_network.train()
        return u.copy()

    # update the network
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        ## TODO: compute and minimize the loss
        self.Q_network.train()          # Set network to trainable
        self.Q_target_network.eval()    # Forward propagation only, no backward propagation

        #shape of output from the model (batch_size, action_dim) = (64, 4)
        predicted_targets = self.Q_network(states).gather(1, actions)
    
        labels_next = self.Q_target_network(next_states).detach().max(1)[0].unsqueeze(1)

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels      = rewards + (gamma*labels_next*(1 - dones))         # Update those not done
        criterion   = torch.nn.MSELoss()                                # Set function to criterion
        loss        = criterion(predicted_targets, labels)              # Loss
        self.Q_optim.zero_grad()                                        # Reset gradient (previous gradient calculated is saved so need to reset)
        loss.backward()                                                 # Backward Propagate
        for param in self.Q_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.Q_optim.step()                                             # Step ++

        # ------------------- update target network ------------------- #
        self._soft_update_target_network()

class ReplayBuffer:
    """Fixed -size buffer to store experience tuples."""
    
    def __init__(self, buffer_size, batch_size):
        """
        Initialize a ReplayBuffer object.
        
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int):  size of each training batch
        """
        
        self.memory         = deque(maxlen = buffer_size)
        self.batch_size     = batch_size
        self.experiences    = namedtuple("Experience", field_names=["state",
                                                                    "action",
                                                                    "reward",
                                                                    "next_state",
                                                                    "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k = self.batch_size)
        states      = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions     = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards     = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones       = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)



epsilon                 = 0.45
max_episode             = 10000

env                     = gym.make("Switch2-v0") # Use "Switch4-v0" for the Switch-4 game
agents                  = [DQNAgent(i) for i in range(env.n_agents)]
ep_reward_mem           = []
training_time_mem       = []
final_agent_pos         = {0: [0.0, 1.0], 1: [0.0, 0.0],
                           2: [1.0, 1.0], 3: [1.0, 0.0]}
n_agent_reached_mem     = []

for episode in range(0, max_episode):
    ep_reward       = 0
    episode_t_step  = 0
    n_agent_reached = 0
    obs_n           = env.reset()
    done_n          = [False]*env.n_agents # [False for _ in range(env.n_agents)]
    while not all(done_n):
        actions = []
        for agent_idx in range(env.n_agents):
            # Append time step:
            obs_n[agent_idx].append(episode_t_step)
            # Collect actions:
            actions.append(agents[agent_idx].select_action(obs_n[agent_idx], epsilon))
        
        # Use action to run the environment:
        next_obs_n, reward_n, done_n, info = env.step(actions)
        for agent_idx in range(env.n_agents):
            # Append time step:
            next_obs_n[agent_idx].append(episode_t_step)
            # Save to replay buffer + learn (QNetwork)
            agents[agent_idx].step(obs_n[agent_idx], actions[agent_idx], sum(reward_n), next_obs_n[agent_idx], done_n[agent_idx])

        # Update observation coordinate:
        for agent_idx in range(env.n_agents):
            obs_n[agent_idx] = next_obs_n[agent_idx][:2]
        
        episode_t_step  += 1
        ep_reward       += sum(reward_n)
        # env.render()
    for agent in range(env.n_agents):
        if final_agent_pos[agent] == next_obs_n[agent][0:2]:
            n_agent_reached += 1

    n_agent_reached_mem.append(n_agent_reached)
    training_time_mem.append(episode_t_step)
    if epsilon > 0.05:
        epsilon -= 0.005
    
    if episode % 1000 == 0:
        print(f"episode: {episode}")

    ep_reward_mem.append(ep_reward)
    env.close()