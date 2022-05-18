from src.environments import ReplaceTokenEnvironment

if __name__ == '__main__':
    env = ReplaceTokenEnvironment()

    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()

        if done:
            #env.reset()
            break

    env.close()