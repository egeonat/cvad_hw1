import numpy as np
import torch
from func_timeout import FunctionTimedOut, func_timeout
from utils.rl_utils import generate_noisy_action_tensor

from .off_policy import BaseOffPolicy


class TD3(BaseOffPolicy):
    def _compute_q_loss(self, data):
        """Compute q loss for given batch of data."""
        # Your code here
        pass

    def _compute_p_loss(self, data):
        """Compute policy loss for given batch of data."""
        # Your code here
        pass

    def _extract_features(self, state):
        """Extract whatever features you wish to give as input to policy and q networks."""
        # Your code here
        pass

    def _take_step(self, state, action):
        try:
            action_dict = {
                "throttle": np.clip(action[0, 0].item(), 0, 1),
                "brake": abs(np.clip(action[0, 0].item(), -1, 0)),
                "steer": np.clip(action[0, 1].item(), -1, 1),
            }
            new_state, reward_dict, is_terminal = func_timeout(
                20, self.env.step, (action_dict,))
        except FunctionTimedOut:
            print("\nEnv.step did not return.")
            raise
        return new_state, reward_dict, is_terminal

    def _collect_data(self, state):
        """Take one step and put data into the replay buffer."""
        features = self._extract_features(state)
        if self.step >= self.config["exploration_steps"]:
            action = self.policy(features, [state["command"]])
            action = generate_noisy_action_tensor(
                action, self.config["action_space"], self.config["policy_noise"], 1.0)
        else:
            action = self._explorer.generate_action(state)
        if self.step <= self.config["augment_steps"]:
            action = self._augmenter.augment_action(action, state)

        # Take step
        new_state, reward_dict, is_terminal = self._take_step(state, action)

        new_features = self._extract_features(state)

        # Prepare everything for storage
        stored_features = [f.detach().cpu().squeeze(0) for f in features]
        stored_command = state["command"]
        stored_action = action.detach().cpu().squeeze(0)
        stored_reward = torch.tensor([reward_dict["reward"]], dtype=torch.float)
        stored_new_features = [f.detach().cpu().squeeze(0) for f in new_features]
        stored_new_command = new_state["command"]
        stored_is_terminal = bool(is_terminal)

        self._replay_buffer.append(
            (stored_features, stored_command, stored_action, stored_reward,
             stored_new_features, stored_new_command, stored_is_terminal)
        )
        self._visualizer.visualize(new_state, stored_action, reward_dict)
        return reward_dict, new_state, is_terminal
