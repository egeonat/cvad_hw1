from abc import ABC, abstractmethod
from os.path import join

import torch
import torch.optim as optim
from func_timeout import FunctionTimedOut, func_timeout
from models.policy import MultiLayerPolicy
from models.q import MultiLayerQ
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.rl_utils import get_q_params, polyak_update

from rl.tools.augmenter import get_augmenter
from rl.tools.controller import get_controller
from rl.tools.explorer import get_explorer
from rl.tools.replay_buffer import ReplayBuffer
from rl.tools.visualizer import get_visualizer


class BaseOffPolicy(ABC):
    def __init__(self, env, config):
        self.env = env
        self.config = config

        policies, q_nets = self._make_models(config)
        self.policy, self.target_policy = policies
        self.q_nets, self.target_q_nets = q_nets

        # List of parameters for both Q-networks saved for convenience
        self.q_params = get_q_params(self.q_nets)
        # Set up optimizers for policy and q-function
        self.q_optim = optim.Adam(self.q_params, config["q_lr"])
        self.p_optim = optim.Adam(self.policy.parameters(), config["p_lr"])

        # Various values to track
        self.step = 0
        self.episode = 0
        self.num_update = 0
        self.best_eval_return = 0

        self._logger = None
        self._explorer = get_explorer(config)
        self._controller = get_controller(config)
        self._augmenter = get_augmenter(config)
        self._visualizer = get_visualizer(config)
        self._replay_buffer = ReplayBuffer(config["replay_buffer_size"])

    def load_ckpt(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.target_policy.load_state_dict(checkpoint["target_policy_state_dict"])
        for i in range(2):
            self.q_nets[i].load_state_dict(checkpoint["q_state_dicts"][i])
            self.target_q_nets[i].load_state_dict(checkpoint["target_q_state_dicts"][i])
        if "replay_buffer_q" in checkpoint:
            self._replay_buffer.load_data(checkpoint["replay_buffer_q"], checkpoint["step"])
        self.q_optim.load_state_dict(checkpoint["q_optim_state_dict"])
        self.p_optim.load_state_dict(checkpoint["p_optim_state_dict"])
        self.step = checkpoint["step"]
        self.episode = checkpoint["episode"]
        self.num_update = checkpoint["num_update"]
        self.best_eval_return = checkpoint["best_eval_return"]

    def train(self):
        self._logger = SummaryWriter(
            self.config["experiment_name"] + "_logs", purge_step=self.step)

        # Get initial values from environment by reseting
        state, is_terminal = self._reset_env()

        ep_step = 0
        ep_return = {}
        eval_flag = False
        # main loop
        while True:
            # Take one step and add it to replay buffer
            with torch.no_grad():
                r_dict, new_state, is_terminal = self._collect_data(state)

            # Update episode return
            for key, val in r_dict.items():
                ep_return[key] = ep_return.get(key, 0) + self.config["discount"]**ep_step * val

            print("Episode {:03d} - Step {:05d} - Env Step {:04d}".format(
                self.episode, self.step, ep_step), end="\r")
            ep_step += 1
            self.step += 1

            # Critical, easy to overlook step: make sure to update observations!
            state = new_state

            # If last step was the terminal step or we reached max episode steps, reset env.
            if is_terminal or ep_step > self.config["episode_max_steps"]:
                print("\nEpisode length: {} steps.".format(ep_step))
                print("Episode return: {}".format(ep_return["reward"]))
                print("-" * 80)
                self._logger.add_scalar("Episode length", ep_step, self.step)
                self._logger.add_scalars("Episode return", ep_return, self.step)

                self.episode += 1
                ep_step = 0
                ep_return = {}
                # Do evaluation if evaluation_interval was passed during this episode
                if eval_flag:
                    eval_flag = False
                    print("Evaluating target policy")
                    ep_len, ep_return, ep_mean_reward, _ = self.test_agent()
                    self._logger.add_scalar("Eval episode length", ep_len, self.step)
                    self._logger.add_scalars("Eval episode return", ep_return, self.step)
                    self._logger.add_scalars("Eval mean reward", ep_mean_reward, self.step)
                    print("\nEvaluation over")
                    if ep_return["reward"] > self.best_eval_return:
                        self.best_eval_return = ep_return["reward"]
                        print("New best eval return: {}".format(ep_return["reward"]))
                        print("Saving as best checkpoint")
                        self.save_checkpoint(f"best_{self.config['experiment_name']}.ckpt",
                                             include_replay_buffer=False)
                    print("-" * 80)
                state, is_terminal = self._reset_env()

            # Update if it's time
            if self._should_update():
                rb_list = self._replay_buffer.to_list()
                loader = DataLoader(rb_list, self.config["batch_size"],
                                    shuffle=True, pin_memory=True, drop_last=True)
                self._update(loader)
                self.num_update += 1

            # Check whether it is time to test our agent
            if self._should_eval():
                eval_flag = True

            # Save agent if it's time
            if self.step % self.config["save_interval"] == 0 or self.step == 1:
                print("Saving checkpoint" + " " * 60)
                self.save_checkpoint("last_" + self.config["experiment_name"] + ".ckpt")
                print("-" * 80)

        print("Saving checkpoint")
        self.save_checkpoint("last_" + self.config["experiment_name"] + ".ckpt")
        print("-" * 80)

    def test_agent(self, num_trials=5):
        total_ep_step = 0
        total_ep_return = {}
        total_ep_reward = {}
        terminal_histogram = {}
        for _ in range(num_trials):
            ep_step = 0

            # Env.reset() gives the starting state and is_terminal values
            state, is_terminal = self._reset_env()

            while not is_terminal and ep_step < self.config["episode_max_steps"]:
                # Generate action
                features = self._extract_features(state)
                action = self.policy(features, [state["command"]])

                # Take step
                new_state, reward_dict, is_terminal = self._take_step(state, action)

                # Update episode return
                for key, val in reward_dict.items():
                    total_ep_return[key] = (total_ep_return.get(key, 0)
                                            + self.config["discount"] ** ep_step * val)
                    total_ep_reward[key] = total_ep_reward.get(key, 0) + val
                ep_step += 1

                self._visualizer.visualize(
                    new_state, action.detach().cpu().squeeze(0), reward_dict)
                print("Evaluation - Env Step {:04d}".format(ep_step), end="\r")

                # Critical, easy to overlook step: make sure to update observations!
                state = new_state

            total_ep_step += ep_step
            if not is_terminal:
                is_terminal = ["timeout"]
            terminal_histogram[is_terminal[0]] = (terminal_histogram.get(is_terminal[0], 0)+1)

        mean_ep_step = total_ep_step / num_trials
        mean_ep_return = {}
        for key, val in total_ep_return.items():
            mean_ep_return[key] = val / num_trials
            total_ep_reward[key] = total_ep_reward[key] / total_ep_step
        return mean_ep_step, mean_ep_return, total_ep_return, terminal_histogram

    def save_checkpoint(self, ckpt_str, *, include_replay_buffer=True):
        ckpt = {
            "policy_state_dict": self.policy.state_dict(),
            "target_policy_state_dict": self.target_policy.state_dict(),
            "p_optim_state_dict": self.p_optim.state_dict(),
            "q_state_dicts": [q.state_dict() for q in self.q_nets],
            "target_q_state_dicts": [target_q.state_dict() for target_q in self.target_q_nets],
            "q_optim_state_dict": self.q_optim.state_dict(),
            "step": self.step,
            "episode": self.episode + 1,
            "num_update": self.num_update,
            "best_eval_return": self.best_eval_return
        }
        if include_replay_buffer:
            ckpt["replay_buffer_q"] = self._replay_buffer.q
        torch.save(ckpt, join("checkpoints", "rl", ckpt_str))

    def _update(self, loader):
        num_batches = min(len(loader), self.config["update_every"])
        print("\nDoing updates with {} batches...".format(num_batches))

        mean_q_loss = 0
        mean_p_loss = 0
        for i, data in enumerate(loader):
            if i == self.config["update_every"]:
                break
            iter_loss = 0
            self.q_optim.zero_grad()
            self.p_optim.zero_grad()

            # Calculating q_loss and adding to iter loss
            q_val_estimates, q_loss = self._compute_q_loss(data)
            iter_loss += q_loss
            # Logging
            mean_q_loss += q_loss.item() / num_batches

            # Calculate p_loss every 2 updates. Don't start at first update
            if (self.num_update + 1) % self.config["policy_delay"] == 0:
                # Freeze Q-networks so you don't waste computational effort
                # computing gradients for them during the policy learning step.
                for param in self.q_params:
                    param.requires_grad = False

                # Calculating p_loss and adding to iter loss
                p_loss = self._compute_p_loss(data)
                iter_loss += p_loss
                # Logging
                mean_p_loss += p_loss.mean().item() / num_batches

                # Unfreeze Q-networks so you can optimize it at next DDPG step.
                for param in self.q_params:
                    param.requires_grad = True

            iter_loss.backward()
            if self.config["q_clip_grad_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(self.q_nets[0].parameters(),
                                               self.config["q_clip_grad_norm"])
                torch.nn.utils.clip_grad_norm_(self.q_nets[1].parameters(),
                                               self.config["q_clip_grad_norm"])
            self.q_optim.step()
            if (self.num_update + 1) % self.config["policy_delay"] == 0:
                if self.config["p_clip_grad_norm"] is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                                   self.config["p_clip_grad_norm"])
                self.p_optim.step()
                # Update target networks
                with torch.no_grad():
                    for j in range(2):
                        polyak_update(
                            self.q_nets[j], self.target_q_nets[j], self.config["polyak"])
                    polyak_update(
                        self.policy, self.target_policy, self.config["polyak"])
        self._log_updates(mean_q_loss, mean_p_loss)

    def _make_models(self, config):
        """Creates models according to configs, sends them to GPU and returns them."""
        policy = MultiLayerPolicy()
        target_policy = MultiLayerPolicy()
        q_nets = [MultiLayerQ(config) for _ in range(2)]
        target_q_nets = [MultiLayerQ(config) for _ in range(2)]

        # Make target policy and q weights equal to normal policy and q weights
        target_policy.load_state_dict(policy.state_dict())
        for i in range(2):
            target_q_nets[i].load_state_dict(q_nets[i].state_dict())

        # Send models to GPU
        policy.cuda()
        target_policy.cuda()
        for i in range(2):
            q_nets[i].cuda()
            target_q_nets[i].cuda()

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in target_policy.parameters():
            p.requires_grad = False
        for target_q_net in target_q_nets:
            # Make target q weights equal to q weights
            for p in target_q_net.parameters():
                p.requires_grad = False
        return (policy, target_policy), (q_nets, target_q_nets)

    def _log_updates(self, mean_q_loss, mean_p_loss):
        if self.num_update % 10 == 0:
            q_net_params = [[], []]
            q_net_grad_norms = [0.0, 0.0]
            for idx, params in enumerate([q.parameters() for q in self.q_nets]):
                for param in params:
                    q_net_params[idx].append(param.view(-1))
                    if param.grad is not None:
                        q_net_grad_norms[idx] += param.grad.detach().data.norm(2).item()**2
            q_net_grad_norms = [norm ** 0.5 for norm in q_net_grad_norms]
            self._logger.add_scalars("Q network grad norms",
                                     {"q_net[0]": q_net_grad_norms[0],
                                      "q_net[1]": q_net_grad_norms[1]},
                                     self.step)
        self._logger.add_scalar("Qval loss mean", mean_q_loss, self.step)

        if (self.num_update + 1) % self.config["policy_delay"] == 0:
            p_net_params = []
            p_net_grad_norm = 0.0
            num_none_params = 0
            num_valid_params = 0
            for param in self.policy.parameters():
                p_net_params.append(param.view(-1))
                if param.grad is not None:
                    p_net_grad_norm += param.grad.detach().data.norm(2).item()**2
                    num_valid_params += 1
                else:
                    num_none_params += 1
            p_net_grad_norm = p_net_grad_norm ** 0.5
            self._logger.add_scalar("Policy network grad norm", p_net_grad_norm, self.step)
            self._logger.add_scalar("Policy loss mean", mean_p_loss, self.step)
        print("-" * 80)

    def _reset_env(self):
        try:
            state, _, is_terminal = func_timeout(30, self.env.reset)
        except FunctionTimedOut:
            print("\nEnv.reset did not return.")
            raise
        return state, is_terminal

    def _should_update(self):
        should_update = (self.step > self.config["no_update_steps"]
                         and self.step % self.config["update_every"] == 0)
        return should_update

    def _should_eval(self):
        should_eval = (self.step % self.config["evaluation_interval"] == 0
                       and self.step > self.config["no_update_steps"])
        return should_eval

    @abstractmethod
    def _collect_data(self, state):
        """Take one step and put data into the replay buffer."""
        pass

    @abstractmethod
    def _take_step(self, state, action):
        """Takes a step on the environment based on the generated action."""
        pass

    @abstractmethod
    def _compute_q_loss(self, data):
        """Compute q_val estimates and loss for given batch of data."""
        pass

    @abstractmethod
    def _compute_p_loss(self, data):
        """Compute policy loss for given batch of data."""
        pass
