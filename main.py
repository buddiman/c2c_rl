from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo import DEFAULT_CONFIG

from src.environments import ReplaceTokenEnvironment
from src.utils.tokenizer import SimpleTokenizer
from colorama import init

import os
import ray
import ray.rllib.agents.ppo as ppo

if __name__ == '__main__':
    # init colorama
    init()

    # init ray
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    config = DEFAULT_CONFIG.copy()
    agent = PPOTrainer(config, ReplaceTokenEnvironment.ReplaceTokenEnvironment)

    for i in range(20):
        print("Training iteration {}...".format(i))
        agent.train()

    # init and (train) load tokenizer
    tokenizer = SimpleTokenizer()
    # tokenizer.train_and_save([f"data/test/oneliners_no_long_text.data"], "oneliners_no_long_text.data.json")
    tokenizer.load("oneliners_no_long_text.data.json")

    env = ReplaceTokenEnvironment.ReplaceTokenEnvironment(tokenizer)
    env.load_data("data/test/oneliners_no_long_text.data")

    done = False
    max_state = -1
    cumulative_reward = 0

    while not done:
        action = agent.compute_action(state)
        state, reward, done, results = env.step(action)
        max_state = max(max_state, state)
        cumulative_reward += reward
    '''
    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()

        if done:
            if env.finished and env.data_index < len(env.data):
                env.load_new_line()
                env.reset()
            else:
                env.reset()

    env.close()
    '''
