import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from fail.env.block_touching.block_touching import BlockTouching
from fail.env.block_pushing.block_pushing import BlockPushing
#from fail.env.peg_insertion.peg_insertion import PegInsertion
from fail.gym_util.async_vector_env import AsyncVectorEnv
from fail.gym_util.sync_vector_env import SyncVectorEnv
from fail.gym_util.multistep_wrapper import MultiStepWrapper
from fail.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from gym.wrappers import FlattenObservation

from fail.policy.base_policy import BasePolicy
from fail.utils.torch_utils import dict_apply
from fail.env_runner.base_runner import BaseRunner


class DrakeObjectStateRunner(BaseRunner):
    def __init__(
        self,
        output_dir,
        env,
        num_train=10,
        num_train_vis=3,
        train_start_seed=0,
        num_test=22,
        num_test_vis=6,
        test_start_seed=10000,
        max_steps=200,
        num_obs_steps=8,
        num_action_steps=8,
        fps=5,
        crf=22,
        past_action=False,
        abs_action=False,
        tqdm_interval_sec=5.0,
        num_envs=None,
    ):
        super().__init__(output_dir)
        num_envs = num_train + num_test if num_envs is None else num_evs

        task_fps = 10
        steps_per_render = max(10 // fps, 1)

        if env == "block_touching":
            env = BlockTouching
        elif env == "block_pushing":
            env = BlockPushing
        elif env == "peg_insertion":
            env = PegInsertion
        else:
            raise ValueError("Invalid Drake env specified.")

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    FlattenObservation(env()),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=num_obs_steps,
                n_action_steps=num_action_steps,
                max_episode_steps=max_steps,
            )

        env_fns = [env_fn] * num_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(num_train):
            seed = train_start_seed + i
            enable_render = i < num_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", wv.util.generate_id() + ".mp4"
                    )
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("train/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(num_test):
            seed = test_start_seed + i
            enable_render = i < num_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", wv.util.generate_id() + ".mp4"
                    )
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("test/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.num_obs_steps = num_obs_steps
        self.num_action_steps = num_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        num_envs = len(self.env_fns)
        num_inits = len(self.env_init_fn_dills)
        num_chunks = math.ceil(num_inits / num_envs)

        # allocate data
        all_video_paths = [None] * num_inits
        all_rewards = [None] * num_inits
        # last_info = [None] * num_inits

        for chunk_idx in range(num_chunks):
            start = chunk_idx * num_envs
            end = min(num_inits, start + num_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            num_diff = num_envs - len(this_init_fns)
            if num_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == num_envs

            # init envs
            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval DrakeObjectStateRunner {chunk_idx+1}/{num_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )
            done = False
            while not done:
                obs = obs.astype(np.float32)
                # create obs dict
                np_obs_dict = {
                    "robot_state": obs[:, :, :90],
                    "world_state": obs[:, :, 90:],
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict["past_action"] = past_action[
                        :, -(self.num_obs_steps - 1) :
                    ].astype(np.float32)
                # device transfer
                obs_dict = dict_apply(
                    np_obs_dict, lambda x: torch.from_numpy(x).to(device=device)
                )

                # run policy
                with torch.no_grad():
                    action_dict = policy.get_action(
                        obs_dict["robot_state"], obs_dict["world_state"], device
                    )

                # device_transfer
                np_action_dict = dict_apply(
                    action_dict, lambda x: x.detach().to("cpu").numpy()
                )

                action = np_action_dict["action"]

                # step env
                obs, reward, done, _ = env.step(action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call("get_attr", "reward")[
                this_local_slice
            ]
            # last_info[this_global_slice] = [
            #    dict((k, v[-1]) for k, v in x.items()) for x in info
            # ][this_local_slice]

        # log
        total_rewards = collections.defaultdict(list)
        prefix_event_counts = collections.defaultdict(
            lambda: collections.defaultdict(lambda: 0)
        )
        prefix_counts = collections.defaultdict(lambda: 0)

        log_data = dict()
        for i in range(num_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            this_rewards = all_rewards[i]
            total_reward = np.unique(this_rewards).sum()

            total_rewards[prefix].append(total_reward)
            log_data[prefix + f"sim_max_reward_{seed}"] = total_reward

            # aggregate event counts
            prefix_counts[prefix] += 1
            # for key, value in last_info[i].items():
            #    delta_count = 1 if value > 0 else 0
            #    prefix_event_counts[prefix][key] += delta_count

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix + f"sim_video_{seed}"] = sim_video

        # log aggregate metrics
        for prefix, value in total_rewards.items():
            name = prefix + "mean_score"
            value = np.mean(value)
            log_data[name] = value

        # summarize probabilities
        for prefix, events in prefix_event_counts.items():
            prefix_count = prefix_counts[prefix]
            for event, count in events.items():
                prob = count / prefix_count
                key = prefix + event
                log_data[key] = prob

        return log_data
