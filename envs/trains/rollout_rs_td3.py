import argparse
import numpy as np

from rljax.algorithm import RSFlowTD3, RS_IQNSAC, RS_IQNTD3
from envs.monotone_test_env import TestEnv
from tqdm import tqdm


algo_map = {"RSFlowTD3": RSFlowTD3, "RS_IQNSAC": RS_IQNSAC, "RS_IQNTD3": RS_IQNTD3}


def run(args):
    env = TestEnv()
    env.seed(2 ** 31 - args.seed)
    algo = algo_map[args.algo](
        num_agent_steps=args.num_agent_steps,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=args.seed,
        num_critics=args.num_critics,
        num_quantiles_to_drop=args.num_quantiles_to_drop,
    )
    algo.load_params(args.save_dir)
    scores = []
    actions = []
    for _ in tqdm(range(args.num_eval)):
        done = False
        obs = env.reset()
        score = 0
        while not done:
            action = algo.select_action(obs)
            actions.append(action.item())
            obs, reward, done, info = env.step(action)
            score += reward
        scores.append(score)
    scores = np.asarray(scores)
    actions = np.asarray(actions)
    np.savez(args.scores_save_dir, **{"scores": scores, "actions": actions})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", type=str, default='RS_IQNTD3')
    p.add_argument("--save_dir", type=str, default="/home/yoo/PycharmProjects/rljax/envs/trains/logs/test/RS_IQNTD3-seed0-cvar:0.5-20221108-1604/param/step100000")
    p.add_argument("--num_quantiles_to_drop", type=int, default=2)
    p.add_argument("--num_critics", type=int, default=2)
    p.add_argument("--num_eval", type=int, default=10000)
    p.add_argument("--num_agent_steps", type=int, default=1)
    p.add_argument("--scores_save_dir", type=str, default='TD3_simple_score_cvar_0.5')
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)

