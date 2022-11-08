import argparse
import numpy as np

from rljax.algorithm import RCDSAC
from envs.monotone_test_env import TestEnv
from tqdm import tqdm


def run(args):
    env = TestEnv()
    env.seed(2 ** 31 - args.seed)
    algo = RCDSAC(
        num_agent_steps=args.num_agent_steps,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=args.seed,
        num_critics=args.num_critics,
        num_quantiles_to_drop=args.num_quantiles_to_drop,
        target_confidence=args.target_risk
    )
    algo.load_params(args.save_dir)
    scores = []
    actions = []
    for _ in tqdm(range(args.num_eval)):
        done = False
        obs = env.reset()
        score = 0
        while not done:
            action = algo.select_action_with_target(obs, args.target_risk)
            actions.append(action.item())
            obs, reward, done, info = env.step(action)
            score += reward
        scores.append(score)
    scores = np.asarray(scores)
    actions = np.asarray(actions)
    np.savez(args.scores_save_dir, **{"scores": scores, "actions": actions})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", type=str, default="/home/yoo/PycharmProjects/rljax/envs/trains/logs/test/RCDSAC-seed2-20221108-0543/param/step300000")
    p.add_argument("--num_quantiles_to_drop", type=int, default=4)
    p.add_argument("--num_critics", type=int, default=2)
    p.add_argument("--num_eval", type=int, default=10000)
    p.add_argument("--num_agent_steps", type=int, default=1)
    p.add_argument("--scores_save_dir", type=str, default='simple_score_env')
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--target_risk", type=float, default=0.5)
    for i in [0, 0.5, 1]:
        args = p.parse_args()
        args.target_risk = i
        args.scores_save_dir = 'score_{}'.format(i)
        run(args)

