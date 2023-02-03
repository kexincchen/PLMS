class RandomSolver(object):

    @classmethod
    def solve(cls, env, episodes = 10):
        scores = []
        for episode in range(0,episodes):
            state = env.reset()
            done = False
            score = 0

            while not done:
                action = env.action_space.sample()
                n_state, reward, done, info = env.step(action)
                score+=reward
            print('Episode:{} Score:{}'.format(episode, score))
            scores.append(score)
        return scores