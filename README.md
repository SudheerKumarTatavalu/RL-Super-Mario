
# Train a Mario-playing RL Agent

```
%%bash
pip install gym-super-mario-bros==7.4.0
pip install tensordict==0.2.0
pip install torchrl==0.2.0
```
```
import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os

```
# Gym is an OpenAI toolkit for RL
```
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
```

# NES Emulator for OpenAI Gym
```
from nes_py.wrappers import JoypadSpace
```

# Super Mario environment for OpenAI Gym
```
import gym_super_mario_bros
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
```

## RL Definitions

**Environment:** The setting in which an agent operates, providing the context for the agent's actions and learning experiences.

**Action:** Denoted as \(a\), actions represent the decisions or responses made by the agent within the environment. The entire set of possible actions is known as the "action-space."

**State:** Denoted as \(s\), the current representation of the environment. States capture the essential characteristics or features defining the environment at a given moment. The collection of all possible states is referred to as the "state-space."

**Reward:** Denoted as \(r\), rewards serve as crucial feedback from the environment to the agent. Rewards drive the learning process by indicating the desirability of an agent's actions. The accumulation of rewards over multiple time steps is termed the "Return."

**Optimal Action-Value Function:** Represented as \(Q^*(s,a)\), this function provides the expected return when starting in state \(s\), taking a specific action \(a\), and subsequently choosing actions that maximize returns in future time steps. The symbol \(Q\) can be interpreted as the "quality" of an action within a given state. The goal is to approximate this function to guide optimal decision-making.

## Environment

### Initialize Environment

In the Mario environment, various elements such as tubes, mushrooms, and other components collectively form the surroundings in which Mario operates.

As Mario takes actions, the environment dynamically reacts by transitioning to a new state, providing a corresponding reward, and offering additional information. Each action taken by Mario influences the state of the game, leading to alterations in the arrangement of tubes, the appearance of mushrooms, and the interactions with other in-game components. This ongoing exchange between Mario's actions and the environment's responses shapes the overall gameplay experience, with rewards serving as feedback that guides Mario's learning process and decision-making in the game world.
"""

# Initialize Super Mario environment 
```
if gym.__version__ < '0.26':
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
else:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='rgb', apply_api_compatibility=True)
```

# Limit the action-space to
###   0. walk right
###   1. jump right
```
env = JoypadSpace(env, [["right"], ["right", "A"]])

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")
```

### Preprocess Environment

In the Mario environment, the information returned to the agent is encapsulated in the `next_state` variable. However, the default representation of each state is a sizable array with dimensions `[3, 240, 256]`, providing more details than necessary for our agent's decision-making process. Notably, certain elements such as the color of pipes or the sky are irrelevant to Mario's actions.

To streamline the state representation, we leverage **Wrappers**—customizable preprocessing modules that transform the environment data before presenting it to the agent.

The `GrayScaleObservation` wrapper is a common choice for converting RGB images to grayscale. This transformation retains essential information while significantly reducing the state size to `[1, 240, 256]`.

Similarly, the `ResizeObservation` wrapper downsamples each observation, producing a compact square image with dimensions `[1, 84, 84]`.

The `SkipFrame` wrapper, implemented as a custom class inheriting from `gym.Wrapper`, optimizes the processing of consecutive frames. Recognizing that consecutive frames often exhibit minimal variation, this wrapper allows us to skip a defined number of intermediate frames. The n-th frame aggregates rewards accumulated during each skipped frame, preserving vital information while enhancing computational efficiency.
  
Finally, the `FrameStack` wrapper consolidates consecutive frames into a single observation point. This compression enables the model to identify Mario's actions, such as landing or jumping, by examining the directional changes in his movement across several previous frames.



```
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation
```

# Apply Wrappers to environment
```
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)
```

Following the application of the previously mentioned wrappers to the environment, the resulting wrapped state is a concatenation of four consecutive grayscale frames. This arrangement is depicted in the left image above. Whenever Mario executes an action, the environment reacts by providing a state structured in this manner.

The structure is embodied as a 3-D array with dimensions `[4, 84, 84]`. Each element in the array corresponds to a specific frame, and the stacking of these frames allows the model to capture temporal information over multiple time steps. This condensed representation serves as the input to the agent's learning model, enabling it to discern patterns and dynamics in Mario's interactions with the environment across successive frames.

## Agent

We create a class ``Mario`` to represent our agent in the game. Mario
should be able to:

-  **Act** according to the optimal action policy based on the current
   state (of the environment).

-  **Remember** experiences. Experience = (current state, current
   action, reward, next state). Mario *caches* and later *recalls* his
   experiences to update his action policy.

-  **Learn** a better action policy over time

```

class Mario:
    def __init__():
        pass

    def act(self, state):
        """Given a state, choose an epsilon-greedy action"""
        pass

    def cache(self, experience):
        """Add the experience to memory"""
        pass

    def recall(self):
        """Sample experiences from memory"""
        pass

    def learn(self):
        """Update online action value (Q) function with a batch of experiences"""
        pass
```

In the following sections, we will populate Mario’s parameters and
define his functions.

### Act

When faced with a particular state, the agent faces a crucial decision—whether to pursue the most optimal action (**exploit**) or opt for a random action (**explore**).

In the case of Mario, there exists a probability defined by `self.exploration_rate` that governs his inclination to explore. In instances where exploration is favored, Mario randomly selects an action. Conversely, when he decides to exploit, he turns to the `MarioNet` model (elaborated in the **Learn** section) to determine the most optimal action. This interplay between exploration and exploitation strategies guides Mario's decision-making process, striking a balance between discovering new possibilities and leveraging learned knowledge for optimal actions in the given state.

```
class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5  # no. of experiences between saving Mario Net

    def act(self, state):
        """
    Given a state, choose an epsilon-greedy action and update value of step.

    Inputs:
    state(``LazyFrame``): A single observation of the current state, dimension is (state_dim)
    Outputs:
    ``action_idx`` (``int``): An integer representing which action Mario will perform
    """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx
```

### Cache and Recall

The functions `cache()` and `recall()` play integral roles in Mario's memory and learning processes.

- **`cache()`:** This function is responsible for storing Mario's experiences in memory. Each time Mario executes an action, he captures the relevant details of the interaction, including the current *state*, the *action* undertaken, the *reward* obtained from the action, the *next state*, and an indicator of whether the game is *done*. By accumulating these experiences, Mario builds a repository of information that forms the basis for his learning.

- **`recall()`:** In this function, Mario engages in a form of retrospective learning. He randomly selects a batch of experiences from his memory. This batch, comprised of multiple instances of state-action-reward transitions, serves as a dataset for Mario's learning process. By drawing on a diverse set of past experiences, Mario gains insights into the dynamics of the game, facilitating the refinement of his decision-making strategies.



```

class Mario(Mario):  # subclassing for continuity
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.batch_size = 32

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (``LazyFrame``),
        next_state (``LazyFrame``),
        action (``int``),
        reward (``float``),
        done(``bool``))
        """
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        # self.memory.append((state, next_state, action, reward, done,))
        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

```
### Learn

Mario employs the DDQN algorithm, which relies on two ConvNets known as $Q_{online}$ and $Q_{target}$ to independently estimate the optimal action-value function.

In our implementation, we utilize a shared feature generator called "features" for both $Q_{online}$ and $Q_{target}$. However, we maintain separate fully connected (FC) classifiers for each. The parameters $\theta_{target}$, corresponding to $Q_{target}$, remain frozen to avoid updates through backpropagation. Instead, these parameters are periodically synchronized with $\theta_{online}$ a process I'll delve into later.

This setup ensures that both networks benefit from a common feature foundation while allowing for individualized learning through distinct FC classifiers. The freezing of $\theta_{target}$ and periodic syncing with $\theta_{online}$ contribute to the stability and effectiveness of the learning process.

### Neural Network

```

class MarioNet(nn.Module):
    """mini CNN structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = self.__build_cnn(c, output_dim)

        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
```

### TD Estimate & TD Target

In the learning process, we consider two crucial values: the **TD Estimate** and the **TD Target**.

The **TD Estimate** represents the anticipated optimal action-value, denoted as ${TD}_e$, for a specific state $s$. It is calculated using $Q_{online}^*(s, a)$, where $a$ is the action.

On the other hand, the **TD Target** is a combination of the current reward and the estimated optimal action-value in the next state $s'$. To determine the action $a'$ in the next state, we select the one that maximizes $Q_{online}(s', a)$. The formula for ${TD}_t$ is then expressed as ${TD}_t = r + \gamma Q_{target}^*(s', a')$, where $r$ is the current reward, $\gamma$ is the discount factor, and $a'$ is the action that maximizes $Q_{online}$ in the subsequent state $s'$.

It's worth noting that since we do not have knowledge of the exact next action $a'$, we use the action that maximizes $Q_{online}$ in the next state $s'$ as a substitute. This approach ensures that our learning process remains practical and effective, incorporating the best available estimate for the next action in the absence of precise information.





```

class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.gamma = 0.9

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
```
### Updating the model

As Mario samples inputs from his replay buffer, we compute $TD_t$
and $TD_e$ and backpropagate this loss down $Q_{online}$ to
update its parameters $\theta_{online}$ ($\alpha$ is the
learning rate ``lr`` passed to the ``optimizer``)

\begin{align}\theta_{online} \leftarrow \theta_{online} + \alpha \nabla(TD_e - TD_t)\end{align}

$\theta_{target}$ does not update through backpropagation.
Instead, we periodically copy $\theta_{online}$ to
$\theta_{target}$

\begin{align}\theta_{target} \leftarrow \theta_{online}\end{align}




```

class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

```
### Save checkpoint

```

class Mario(Mario):
    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

```
### Putting it all together

```

class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)
```
### Logging

```

import numpy as np
import time, datetime
import matplotlib.pyplot as plt


class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards"]:
            plt.clf()
            plt.plot(getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}")
            plt.legend()
            plt.savefig(getattr(self, f"{metric}_plot"))

```
### Let’s play!

While in this specific instance we execute the training loop for a modest 40 episodes, it is strongly recommended to extend the training duration significantly for Mario to thoroughly grasp the intricacies of his environment. To achieve a more profound understanding and mastery of the world, it is advisable to run the training loop for a substantially larger number of episodes, ideally exceeding 40,000. This extensive training duration allows Mario to accumulate diverse experiences, refine his decision-making processes, and ultimately acquire a more comprehensive and nuanced knowledge of the environment he navigates.

```
save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

logger = MetricLogger(save_dir)

episodes = 40
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, trunc, info = env.step(action)

        # Remember
        mario.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = mario.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done or info["flag_get"]:
            break

    logger.log_episode()

    if (e % 20 == 0) or (e == episodes - 1):
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
```
![image](https://github.com/Sudheertatavalu/RL-Super-Mario/assets/150185391/fafc3251-e815-4652-bf4b-065f89d7b89d)

