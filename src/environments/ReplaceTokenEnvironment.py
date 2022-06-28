import random
from operator import itemgetter
from typing import Optional, Union

import numpy as np
import gym
from colorama import Fore, Back, Style
from gym import spaces
#from gym.core import ObsType

from src.utils.tokenizer import SimpleTokenizer

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

    def __init__(self, tokenizer: SimpleTokenizer):
        # Define vocabulary
        self.tokenizer = tokenizer
        self.vocabulary = self.tokenizer.get_vocabulary()
        self.sorted_vocabulary = sorted(self.vocabulary.items(), key=itemgetter(1))
        self.vocabulary_size = self.tokenizer.get_vocabulary_size()

        # data
        self.data = []
        self.data_index = 0

        # finished indicator
        self.finished = False

        # Code line
        self.masked_line = ""
        self.unmasked_line = ""
        self.tokenized_line_unmasked = []
        self.tokenized_line_masked = []
        self.masked_token_position = 0

        # Define actions
        self.actions = ["REPLACE", "MOVE_LEFT", "MOVE_RIGHT"]

        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.actions))

        # Define observation space
        #self.observation_space = spaces.Tuple((spaces.Discrete(self.vocabulary_size), spaces.Discrete(100)))
        self.observation_space = spaces.Dict({
            "tokens": spaces.Box(low=0, high=100, shape=(1, 3), dtype=int),
            "position": spaces.Discrete(100)
        })

        # Define current position
        self.current_position = 0

        # Define current action
        self.current_action = None
        self.forced_next_action = -1

    def render(self, mode="ansi"): #mode="human"):
        print(self.create_terminal_string())
        print("Action: " + self.actions[self.current_action])

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        print(Fore.RED + "Resetting environment..." + Style.RESET_ALL)
        self.current_position = 0
        self.finished = False

    def step(self, action):
        """
        Perform one step in the environment.
        :param action:
        :return:
        """
        reward = 0.0
        done = False

        if self.forced_next_action != -1:
            action = self.forced_next_action
            self.forced_next_action = -1
        else:
            self.current_action = action

        # Handle actions
        if action == 0:
            # Replace
            reward = self.replace()
        elif action == 1:
            # Move left
            reward = self.move_left()
        elif action == 2:
            # Move right
            reward = self.move_right()

        # Check if done
        if reward == REWARD_GOOD_END or reward == REWARD_BAD_END:
            done = True

        #input()

        #return (self.tokenized_line_masked, self.current_position), reward, done, {}
        #return {self.tokenized_line_masked, self.current_position}, reward, done, {}
        return {"tokens": self.tokenized_line_masked, "position": self.current_position}, reward, done, {}

    def replace(self):
        """
        Replace the current token.
        :return:
        """
        reward = 0.0
        if self.current_position != self.masked_token_position:
            reward = REWARD_BAD_END
        else:
            random_token = self.sorted_vocabulary[random.randint(0, self.vocabulary_size - 1)]
            self.tokenized_line_masked[self.current_position] = random_token[1] # 0 = token, 1 = id
            if self.tokenized_line_unmasked.ids[self.current_position] == self.tokenized_line_masked[self.current_position]:
                reward = REWARD_GOOD_END
                self.finished = True
            else:
                reward = REWARD_BAD_SMALL
                self.forced_next_action = 0
        return reward

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
        if self.current_position < len(self.tokenized_line_masked):
            self.current_position += 1
            reward = REWARD_GOOD_NORMAL
        else:
            reward = REWARD_BAD_END  # Never go right when index = max
        return reward

    def load_data(self, dataset):
        with open(dataset) as file:
            self.data = file.readlines()
        self.load_new_line()

    def load_new_line(self):
        if self.data_index < len(self.data):
            self.unmasked_line = self.data[self.data_index]
            self.masked_line = self.data[self.data_index]
            self.tokenized_line_unmasked = self.tokenizer.encode(self.masked_line)
            self.tokenized_line_masked = self.mask_random_token(self.tokenized_line_unmasked.ids)
            self.data_index += 1

    def mask_random_token(self, tokenized_line):
        # MASK token id = 0
        temp_tokenized_line = tokenized_line
        rnd = random.randint(0, len(temp_tokenized_line) - 1)
        temp_tokenized_line[rnd] = 0
        self.masked_token_position = rnd
        return temp_tokenized_line

    def create_terminal_string(self):
        s = ""
        for idx, token in enumerate(self.tokenized_line_masked):
            if idx == self.current_position:
                s += Back.CYAN

            if token == 0:
                s += Fore.RED + " MASK " + Style.RESET_ALL
            else:
                s += " " + self.sorted_vocabulary[token][0] + " " + Style.RESET_ALL
        return s


