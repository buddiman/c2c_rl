from src.environments import ReplaceTokenEnvironment
from src.utils.tokenizer import SimpleTokenizer
from colorama import init

import os

if __name__ == '__main__':
    # init colorama
    init()

    tokenizer = SimpleTokenizer()
    #tokenizer.train_and_save([f"data/test/oneliners.data"], "oneliners.data.json")

    tokenizer.load("oneliners.data.json")

    env = ReplaceTokenEnvironment.ReplaceTokenEnvironment(tokenizer)
    env.load_data("data/test/oneliners.data")

    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()

        if done:
            env.reset()

    env.close()