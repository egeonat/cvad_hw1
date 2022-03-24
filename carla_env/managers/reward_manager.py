class RewardManager():
    """Computes and returns rewards based on states and actions."""
    def __init__(self):
        pass

    def get_reward(self, state, action):
        """Returns the reward as a dictionary. You can include different sub-rewards in the
        dictionary for plotting/logging purposes, but only the 'reward' key is used for the
        actual RL algorithm, which is generated from the sum of all other rewards."""
        reward_dict = {}
        # Your code here

        # Your code here
        reward = 0.0
        for val in reward_dict.values():
            reward += val
        reward_dict["reward"] = reward
        return reward_dict
