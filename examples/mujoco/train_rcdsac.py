import argparse
import os
from datetime import datetime

from rljax.algorithm import RCDSAC
from rljax.env import make_continuous_env
from rljax.trainer import Trainer


def run(args):
    env = make_continuous_env(args.env_id)
    env_test = make_continuous_env(args.env_id)

    algo = RCDSAC(
        num_agent_steps=args.num_agent_steps,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=args.seed,
        num_critics=args.num_critics,
        num_quantiles_to_drop=args.num_quantiles_to_drop,
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join("logs", args.env_id, f"{str(algo)}-seed{args.seed}-{time}")

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_agent_steps=args.num_agent_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
    )
    trainer.train()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="HalfCheetah-v2")
    p.add_argument("--num_quantiles_to_drop", type=int, default=2)
    p.add_argument("--num_critics", type=int, default=2)

    p.add_argument("--num_agent_steps", type=int, default=3 * 10 ** 6)
    p.add_argument("--eval_interval", type=int, default=10000)
    for i in range(3):
        p.add_argument("--seed", type=int, default=i)
        args = p.parse_args()
        args.seed = i
        run(args)
