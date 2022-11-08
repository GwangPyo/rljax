import argparse
import os
from datetime import datetime

from rljax.algorithm import RS_IQNTD3
from envs.monotone_test_env import TestEnv
from rljax.trainer import Trainer


def run(args):
    env = TestEnv()
    env_test = TestEnv()

    algo = RS_IQNTD3(
        num_agent_steps=args.num_agent_steps,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=args.seed,
        num_critics=args.num_critics,
        num_quantiles_to_drop=args.num_quantiles_to_drop,
        risk_measure=args.risk_measure,
        risk_param=args.risk_param,
        tau=5e-3,
        lr_actor=1e-3,
        lr_critic=1e-3,
    )
    algo.verbose = True

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join("logs", args.env_id, f"{str(algo)}-seed{args.seed}-{args.risk_measure}:{args.risk_param}-{time}")

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
    p.add_argument("--env_id", type=str, default="test")
    p.add_argument("--num_quantiles_to_drop", type=int, default=4)
    p.add_argument("--num_critics", type=int, default=2)
    p.add_argument("--num_quantiles", type=int, default=32)
    p.add_argument("--num_agent_steps", type=int, default= 10 ** 5)
    p.add_argument("--eval_interval", type=int, default=10000)
    p.add_argument("--risk_measure", type=str, default='cvar')
    p.add_argument("--risk_param", type=float, default=0)

    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    for i in [0, 0.5, 1.]:
        args = p.parse_args()
        args.risk_param=i
        run(args)
