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
from        torch.distributions     import  Categorical
from        ma_gym.wrappers         import  Monitor
from        IPython.display         import  HTML
from        IPython                 import  display             as  ipythondisplay
from        pyvirtualdisplay        import  Display
from        tqdm                    import  tqdm
from        collections             import  namedtuple, deque


class ActorCriticNetwork(nn.Module):
    def __init__(self, observation_size, action_size):
        super(ActorCriticNetwork, self).__init__()

        # actor network
        self.action_network = nn.Sequential(
                nn.Linear(observation_size, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_size),
                nn.Softmax(dim=-1))

        # critic network
        self.critic_network = nn.Sequential(
                nn.Linear(observation_size, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1))

    def action(self, state):
        action_values               = self.action_network(state)
        action_cat_distri           = Categorical(action_values)
        action                      = action_cat_distri.sample()
        action_log_probability      = action_cat_distri.log_prob(action)

        return action, action_log_probability
    
    def evaluate(self, state, action):
        action_values           = self.action_network(state)
        state_value             = self.critic_network(state)

        action_cat_distri       = Categorical(action_values)
        action_log_probability  = action_cat_distri.log_prob(action)
        action_entropy          = action_cat_distri.entropy()

        return state_value, action_log_probability, action_entropy

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
        self.buffer_size    = buffer_size
        self.batch_size     = batch_size
        self.memory         = deque(maxlen = self.buffer_size)
        self.experiences    = namedtuple("Experience", field_names=["state",
                                                                    "action",
                                                                    "action_log_prob",
                                                                    "reward",
                                                                    "next_state",
                                                                    "done"])
    
    def add(self, state, action, action_log_prob, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experiences(state, action, action_log_prob, reward, next_state, done)
        self.memory.append(e)
    
    def clear(self):
        del self.memory
        self.memory         = deque(maxlen = self.buffer_size)

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        # experiences         = random.sample(self.memory, k = self.batch_size)
        experiences         = self.memory
        states              = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions             = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        action_log_probs    = torch.from_numpy(np.vstack([e.action_log_prob for e in experiences if e is not None])).float()
        rewards             = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states         = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones               = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
        
        return (states, actions, action_log_probs, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class PPOAgent:
    def __init__(self, agent_id, observation_size=3, action_size=5, LR=2e-3, betas=(0.9, 0.99), gamma=0.99, K_epochs=4, eps_clip=0.2):
        # Agent parameters:
        self.agent_id           = agent_id
        self.observation_size   = observation_size
        self.action_size        = action_size
        self.action_step        = 0

        # PPO parameters:
        self.LR                 = LR
        self.UPDATE_EVERY       = 2000
        self.betas              = betas
        self.gamma              = gamma
        self.eps_clip           = eps_clip
        self.K_epochs           = K_epochs
        
        # Memory parameters:
        self.buffer_size        = 10000
        self.batch_size         = 64

        # Memory and Networks:
        self.replay_memory              = ReplayBuffer(self.buffer_size, self.batch_size)   # buffer_size: size of memory --> number of experience, 
                                                                                            # batch_size:  number of experience from the memory used for learning
        self.Loss_MSE                   = nn.MSELoss()
        self.policy_network             = ActorCriticNetwork(observation_size, action_size)
        self.policy_network_optimizer   = optim.Adam(self.policy_network.parameters(), lr=self.LR, betas=self.betas)
        self.old_policy_network         = ActorCriticNetwork(observation_size, action_size)
        self.old_policy_network.load_state_dict(self.policy_network.state_dict())

    def select_action(self, state):
        with torch.no_grad():
            inputs                          = torch.tensor(state, dtype=torch.float32)
            action, action_log_probability  = self.old_policy_network.action(inputs)
        # action = self.Q_network.forward(inputs)[0].detach().numpy()
        return action, action_log_probability

    def step(self, state, action, action_log_prob, reward, next_state, done):
        # Save experience in replay memory
        self.replay_memory.add(state, action, action_log_prob, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.action_step = (self.action_step + 1) % self.UPDATE_EVERY
        if self.action_step == 0:
            experiences = self.replay_memory.sample()
            self.learn(experiences)

    # update the network
    def learn(self, experiences):
        ex_states, ex_actions, ex_action_log_prob, ex_rewards, ex_next_states, ex_dones = experiences

        # Discounted reward:
        discounted_rewards = []
        dis_reward         = 0
        for reward, done in zip(reversed(ex_rewards), reversed(ex_dones)):
            if done:
                dis_reward = 0
            dis_reward = reward + (self.gamma*dis_reward)
            discounted_rewards.insert(0, dis_reward.item())

        # Normalize rewards: (X - mu) / sigma
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-10)
        
        # Update network for K_epochs:
        for epoch in range(self.K_epochs):
            # Evaluate experienced actions:
            state_value, action_log_probability, action_entropy = self.policy_network.evaluate(ex_states, ex_actions.squeeze())
            
            # New vs. old policy ratio: exp(log(pi) - log(pi_old)) = pi/pi_old
            ratios      = torch.exp(action_log_probability - ex_action_log_prob.squeeze())

            # Advantage estimates A: Estimate of the relative value of the selected actions
            # Discounted reward - baseline state estimate
            A           = discounted_rewards - state_value.squeeze().detach()

            # TRPO Surrogate Losses:
            # Normal policy gradients objective
            L1          = ratios*A
            # Clipped version of normal policy gradients objective
            L2          = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip)*A
            loss        = -torch.min(L1, L2) + 0.5*self.Loss_MSE(state_value, discounted_rewards) - 0.01*action_entropy
            PPO_loss    = loss.mean()

            # Back propagate:
            self.policy_network_optimizer.zero_grad()
            PPO_loss.backward()
            self.policy_network_optimizer.step()
        
        # Share network parameters
        self.old_policy_network.load_state_dict(self.policy_network.state_dict())

        # Clear Memory
        self.replay_memory.clear()



max_episode             = 100000
episode                 = 1
train                   = True
mean_scope              = 20

env                     = gym.make("Switch4-v0") # Use "Switch4-v0" for the Switch-4 game
agents                  = [PPOAgent(i) for i in range(env.n_agents)]
final_agent_pos         = {0: [0.0, 1.0], 1: [0.0, 0.0],
                           2: [1.0, 1.0], 3: [1.0, 0.0]}
solving_thershold       = 3.5*env.n_agents
ep_reward_mem           = []
training_time_mem       = []
n_agent_reached_mem     = []

# for episode in range(0, max_episode):
while train and episode < max_episode:
    ep_reward       = 0
    episode_t_step  = 0
    n_agent_reached = 0
    obs_n           = env.reset()
    done_n          = [False]*env.n_agents # [False for _ in range(env.n_agents)]

    while not all(done_n):
        actions                  = []
        action_log_probabilities = []

        for agent_idx in range(env.n_agents):
            # Append time step:
            obs_n[agent_idx].append(episode_t_step)
            # Collect actions:
            action, action_log_probability = agents[agent_idx].select_action(obs_n[agent_idx])
            
            actions.append(action.item())
            action_log_probabilities.append(action_log_probability.detach())
        
        # Use action to run the environment:
        next_obs_n, reward_n, done_n, info = env.step(actions)
        for agent_idx in range(env.n_agents):
            # Append time step:
            next_obs_n[agent_idx].append(episode_t_step)
            # Save to replay buffer + learn (QNetwork)
            agents[agent_idx].step(obs_n[agent_idx], actions[agent_idx], action_log_probabilities[agent_idx], reward_n[agent_idx], next_obs_n[agent_idx], done_n[agent_idx])

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
    
    if episode % 20 == 0:
        print(f"Agents reached: {n_agent_reached} with overall reward {ep_reward} at {episode} episode.", end='\r')

    ep_reward_mem.append(ep_reward)
    if np.mean(ep_reward_mem[-mean_scope:]) > solving_thershold:
        print(f"Done in {episode} episodes.")
        break
    episode += 1
    env.close()


smooth          = 500
reward_list_s   = np.convolve(ep_reward_mem, np.ones((smooth,))/smooth, mode='valid')
plt.figure()
plt.plot(range(len(reward_list_s)), np.array(reward_list_s))
plt.xlabel('Episode')
plt.ylabel('Episode reward')
plt.title("Learning Curve (Reward)")
plt.tight_layout()
plt.show()

training_time = np.convolve(training_time_mem, np.ones((smooth,))/smooth, mode='valid')
plt.figure()
plt.plot(range(len(training_time)), np.array(training_time))
plt.xlabel('Episode')
plt.ylabel('Episode Length')
plt.title("Learning Curve (Time)")
plt.tight_layout()
plt.show()

agents_arrived = np.convolve(n_agent_reached_mem, np.ones((smooth,))/smooth, mode='valid')
plt.figure()
plt.plot(range(len(agents_arrived)), np.array(agents_arrived))
plt.xlabel('Episode')
plt.ylabel('Number of Agents Reaching Target')
plt.title("Agents Arrived")
plt.tight_layout()
plt.show()