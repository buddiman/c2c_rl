import random
import numpy as np
import gym
from gym import spaces

REWARD_LOWER_BOUND = 0
REWARD_UPPER_BOUND = 10

MAX_STEPS = 100


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
        # Define reward space

        self.current_action = None

    def step(self, action):
        """
        Perform one step in the environment.
        :param action:
        :return:
        """
        reward = 0
        done = False

        self.current_action = action

        # Handle actions
        if action == 0:
            # Replace
            pass
        elif action == 1:
            # Keep
            pass
        elif action == 2:
            # Move left
            pass
        elif action == 3:
            # Move right
            pass

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
        pass

    def move_right(self):
        """
        Move the current position to the token to the right.
        :return:
        """
        pass

    def load_vocabulary(self):
        """
        Load the vocabulary from a file.
        :return:
        """
        pass
