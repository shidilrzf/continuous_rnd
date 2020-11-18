from gym.envs.mujoco import HalfCheetahEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector, CustomMDPPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.sac.sac_rnd import SAC_RNDTrainer
from rlkit.torch.sac.sac_rnd_reg import SAC_RNDTrainerReg

from rlkit.torch.networks import FlattenMlp
from rlkit.torch.networks import Mlp

from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


import h5py
import argparse
import os
import gym
import d4rl
import numpy as np
import torch
import time


def load_hdf5(dataset, replay_buffer, max_size):
    all_obs = dataset['observations']
    all_act = dataset['actions']
    N = min(all_obs.shape[0], max_size)

    _obs = all_obs[:N - 1]
    _actions = all_act[:N - 1]
    _next_obs = all_obs[1:]
    _rew = np.squeeze(dataset['rewards'][:N - 1])
    _rew = np.expand_dims(np.squeeze(_rew), axis=-1)
    _done = np.squeeze(dataset['terminals'][:N - 1])
    _done = (np.expand_dims(np.squeeze(_done), axis=-1)).astype(np.int32)

    max_length = 1000
    ctr = 0
    # Only for MuJoCo environments
    # Handle the condition when terminal is not True and trajectory ends due to a timeout
    for idx in range(_obs.shape[0]):
        if ctr >= max_length - 1:
            ctr = 0
        else:
            replay_buffer.add_sample_only(
                _obs[idx], _actions[idx], _rew[idx], _next_obs[idx], _done[idx])
            ctr += 1
            if _done[idx][0]:
                ctr = 0
    ###

    print (replay_buffer._size, replay_buffer._terminals.shape)


def experiment(variant):
    eval_env = gym.make(variant['env_name'])
    expl_env = eval_env
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    ).to(ptu.device)
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    ).to(ptu.device)
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    ).to(ptu.device)
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    ).to(ptu.device)
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    ).to(ptu.device)
    if variant['rnd']:
        rnd_network = Mlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[64, 64],
        ).to(ptu.device)

        rnd_target_network = Mlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[64, 64],
        ).to(ptu.device)
        checkpoint = torch.load(variant['rnd_path'], map_location=map_location)
        rnd_network.load_state_dict(checkpoint['network_state_dict'])
        rnd_target_network.load_state_dict(checkpoint['target_state_dict'])
        print('Loading rnd model: {}'.format(variant['rnd_path']))

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = CustomMDPPathCollector(
        eval_env,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    buffer_filename = None
    if variant['buffer_filename'] is not None:
        buffer_filename = variant['buffer_filename']

    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )

    load_hdf5(eval_env.unwrapped.get_dataset(), replay_buffer,
              max_size=variant['replay_buffer_size'])

    if variant['rnd']:
        if variant['KL']:

            trainer = SAC_RNDTrainerReg(
                env=eval_env,
                policy=policy,
                qf1=qf1,
                qf2=qf2,
                target_qf1=target_qf1,
                target_qf2=target_qf2,
                rnd_network=rnd_network,
                rnd_target_network=rnd_target_network,
                device=ptu.device,
                **variant['trainer_kwargs']
            )
        else:
            trainer = SAC_RNDTrainer(
                env=eval_env,
                policy=policy,
                qf1=qf1,
                qf2=qf2,
                target_qf1=target_qf1,
                target_qf2=target_qf2,
                rnd_network=rnd_network,
                rnd_target_network=rnd_target_network,
                beta=variant['rnd_beta'],
                use_rnd_critic=variant['use_rnd_critic'],
                use_rnd_policy=variant['use_rnd_policy'],
                device=ptu.device,
                **variant['trainer_kwargs']
            )

    else:
        trainer = SACTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **variant['trainer_kwargs']
        )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        batch_rl=True,
        q_learning_alg=True,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='sac_d4rl')
    parser.add_argument("--env", type=str, default='halfcheetah-medium-v0')
    parser.add_argument('--rnd', action='store_true',
                        default=False, help='rnd traning')
    parser.add_argument('--beta', default=5e3, type=float)
    parser.add_argument("--rnd_path", type=str,
                        default='/usr/local/google/home/shideh/')
    parser.add_argument("--rnd_model", type=str,
                        default='Nov-03-2020_1648_halfcheetah-medium-v0.pt')
    parser.add_argument('--rnd_type', default='critic', type=str)
    parser.add_argument('--kl', action='store_true',
                        default=False, help='use bonus in KL regularized way')

    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables cuda (default: False')
    parser.add_argument('--qf_lr', default=3e-4, type=float)
    parser.add_argument('--policy_lr', default=1e-4, type=float)
    parser.add_argument('--num_samples', default=100, type=int)
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--device-id', type=int, default=0,
                        help='GPU device id (default: 0')
    args = parser.parse_args()

    # noinspection
    rnd_path = '{}RL/continuous_rnd/sac/examples/models/{}'.format(
        args.rnd_path, args.rnd_model)
    variant = dict(
        algorithm="SAC",
        rnd=args.rnd,
        rnd_path=rnd_path,
        rnd_beta=args.beta,
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        buffer_filename=args.env,  # halfcheetah_101000.pkl',
        load_buffer=True,
        env_name=args.env,
        seed=args.seed,
        # rnd_type
        use_rnd_policy=False,
        use_rnd_critic=False,
        KL=False,

        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
            num_actions_sample=args.num_samples,

        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=args.policy_lr,
            qf_lr=args.qf_lr,
            reward_scale=1,
            use_automatic_entropy_tuning=True,),
    )

    # timestapms
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    # rnd and the type
    if args.rnd:
        exp_dir = '{}/rnd_{}/{}_{}'.format(args.env,
                                           timestamp, args.rnd_type, args.beta, args.seed)

        if args.rnd_type == 'actor-critic':

            variant["use_rnd_policy"] = True
            variant["use_rnd_critic"] = True

        elif args.rnd_type == 'critic':

            variant["use_rnd_critic"] = True

        else:
            variant["use_rnd_policy"] = True

        if args.kl:
            exp_dir = '{}_kl'.format(exp_dir)
            variant["KL"] = True

        else:
            exp_dir = '{}_{}'.format(exp_dir, args.beta)

    else:
        exp_dir = '{}/offline/{}_{}'.format(args.env, timestamp, args.seed)

    print('experiment dir:logs/{}'.format(exp_dir))
    # setup_logger(exp_name, variant=variant, base_log_dir='logs/')
    setup_logger(variant=variant, log_dir='logs/{}'.format(exp_dir))

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if use_cuda:
        # optionally set the GPU (default=False)
        ptu.set_gpu_mode(True, gpu_id=args.device_id)
        print('using gpu:{}'.format(args.device_id))
        def map_location(storage, loc): return storage.cuda()

    else:
        map_location = 'cpu'
        ptu.set_gpu_mode(False)  # optionally set the GPU (default=False)

    experiment(variant)
