import random
from typing import Optional, Union

import numpy as np
import gym
from gym import spaces
from gym.core import ObsType

REWARD_LOWER_BOUND = 0
REWARD_UPPER_BOUND = 10

MAX_STEPS = 100

# Rewards (make this configurable)
REWARD_GOOD_SMALL = 0.1
REWARD_BAD_SMALL = -0.1
REWARD_GOOD_NORMAL = 1.0
REWARD_BAD_NORMAL = -1.0
REWARD_GOOD_MEDIUM = 3.0
REWARD_BAD_MEDIUM = -3.0
REWARD_GOOD_HIGH = 5.0
REWARD_BAD_HIGH = -5.0
REWARD_GOOD_END = 10.0
REWARD_BAD_END = -10.0


class ReplaceTokenEnvironment(gym.Env):
    """
    A token replace environment.
    """

    def __init__(self):
        # Load Vocabulary

        # Define actions
        self.actions = ["REPLACE", "KEEP", "MOVE_LEFT", "MOVE_RIGHT"]

        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.actions))

        # Define observation space
        self.observation_space = spaces.Box(low=np.array([0]),
                                            high=np.array([100]))  # TODO: real upper boundaries from vocabulary

        # Define current position
        self.current_position = 0

        # Define current action
        self.current_action = None

    def render(self, mode="human"):
        pass

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        self.current_position = 0

    def step(self, action):
        """
        Perform one step in the environment.
        :param action:
        :return:
        """
        reward = 0.0
        done = False

        self.current_action = action

        # Handle actions
        if action == 0:
            # Replace
            reward = 0.0
        elif action == 1:
            # Keep
            reward = 0.0
        elif action == 2:
            # Move left
            reward = self.move_left()
        elif action == 3:
            # Move right
            reward = self.move_right()

        # Check if done
        if reward == REWARD_GOOD_END or reward == REWARD_BAD_END:
            done = True

        return self.current_position, reward, done, {}

    def replace(self):
        """
        Replace the current token.
        :return:
        """
        pass

    def keep(self):
        """
        Keep the current token.
        :return:
        """
        pass

    def move_left(self):
        """
        Move the current position to the token to the left.
        :return:
        """
        reward = 0.0
        if self.current_position > 0:
            self.current_position -= 1
            reward = REWARD_GOOD_SMALL
        else:
            reward = REWARD_BAD_END  # Never go left when index = 0
        return reward

    def move_right(self):
        """
        Move the current position to the token to the right.
        :return:
        """
        reward = 0.0
        if self.current_position < 100000:  # TODO: Set to max length of Line when implemented
            self.current_position += 1
            reward = REWARD_GOOD_NORMAL
        else:
            reward = REWARD_BAD_END  # Never go right when index = max
        return reward

    def load_vocabulary(self):
        """
        Load the vocabulary from a file.
        :return:
        """
        pass
