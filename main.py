from src.environments import ReplaceTokenEnvironment
from src.utils.tokenizer import SimpleTokenizer
from colorama import init

import os
import ray
import ray.rllib.agents.ppo as bandit

if __name__ == '__main__':
    # init colorama
    init()

    # init ray
    ray.shutdown()
    ray.init(ignore_reinit_error = True)

    input("Press any key to start the program") # wait to see Dashboard URL

    # init and (train) load tokenizer
    tokenizer = SimpleTokenizer()
    #tokenizer.train_and_save([f"data/test/oneliners_no_long_text.data"], "oneliners_no_long_text.data.json")
    tokenizer.load("oneliners_no_long_text.data.json")

    env = ReplaceTokenEnvironment.ReplaceTokenEnvironment(tokenizer)
    env.load_data("data/test/oneliners_no_long_text.data")

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