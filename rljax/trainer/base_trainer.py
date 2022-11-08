import os
from datetime import timedelta
from time import sleep
import time as pytime

import pandas as pd
from tqdm import tqdm
from rljax.logger import Logger


class Trainer:
    """
    Trainer.
    """

    def __init__(
        self,
        env,
        env_test,
        algo,
        log_dir,
        seed=0,
        action_repeat=1,
        num_agent_steps=10 ** 6,
        eval_interval=10 ** 4,
        num_eval_episodes=10,
        save_params=True,
    ):
        assert num_agent_steps % action_repeat == 0
        assert eval_interval % action_repeat == 0

        # Envs.
        self.env = env
        self.env_test = env_test

        # Set seeds.
        self.env.seed(seed)
        self.env_test.seed(2 ** 31 - seed)

        # Algorithm.
        self.algo = algo

        # Log setting.
        self.log = {"step": [], "return": []}
        self.csv_path = os.path.join(log_dir, "log.csv")
        self.param_dir = os.path.join(log_dir, "param")
        self.logger = Logger(folder=os.path.join(log_dir, "summary"), output_formats=['stdout', 'tensorboard'])
        # Other parameters.
        self.action_repeat = action_repeat
        self.num_agent_steps = num_agent_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        self.save_params = save_params

    def train(self):
        # Time to start training.
        self.start_time = pytime.time()
        # Initialize the environment.
        state = self.env.reset()

        for step in range(1, self.num_agent_steps + 1):
            state = self.algo.step(self.env, state, self.logger)

            if self.algo.is_update():
                self.algo.update(self.logger)

            if step % self.eval_interval == 0:

                if self.logger:

                    fps = step / (pytime.time() - self.start_time)
                    remaining_steps = (self.num_agent_steps - step)
                    eta_second = int(remaining_steps / fps)
                    self.logger.record("time/fps", int(fps), exclude="tensorboard")
                    self.logger.record("time/eta", timedelta(seconds=eta_second), exclude="tensorboard")
                self.evaluate(step)
                if self.save_params:
                    self.algo.save_params(os.path.join(self.param_dir, f"step{step}"))

        # Wait for the logging to be finished.

    def evaluate(self, step):
        total_return = 0.0
        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset()
            done = False
            while not done:
                action = self.algo.select_action(state)
                state, reward, done, _ = self.env_test.step(action)
                total_return += reward

        # Log mean return.
        mean_return = total_return / self.num_eval_episodes
        if self.logger:
            # To TensorBoard.
            self.logger.record("return/test", mean_return)
            # To CSV.
            self.logger.record("time/step", step * self.action_repeat)
            self.logger.record("time/time_elapsed", self.time, exclude="tensorboard")
            # Log to standard output.
            self.logger.dump(step=step * self.action_repeat)

        pd.DataFrame(self.log).to_csv(self.csv_path, index=False)
    @property
    def time(self):
        return str(timedelta(seconds=int(pytime.time() - self.start_time)))
