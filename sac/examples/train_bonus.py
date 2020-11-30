from gym.envs.mujoco import HalfCheetahEnv
from rlkit.torch.networks import Mlp

import gym
import d4rl
import numpy as np

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.data import TensorDataset, DataLoader


from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import time


def get_random_actions(obs, act, num_random):
    random_actions = torch.FloatTensor(obs.shape[0] * num_random, act.shape[-1]).uniform_(-1, 1)
    action_shape = random_actions.shape[0]
    obs_shape = obs.shape[0]
    num_repeat = int(action_shape / obs_shape)
    obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
    data = torch.cat((obs_temp, random_actions), dim=1)
    return data


def train(network, dataloader, optimizer, epoch, device):

    loss_func = nn.BCELoss(reduction='sum')

    network.train()
    # desc = 'Train'

    # tqdm_bar = tqdm(dataloader)
    batch_loss = 0
    for batch_idx, (obs, act) in enumerate(dataloader):
        batch_size = obs.size(0)

        obs = obs.to(device)
        act = act.to(device)

        y_ones = torch.ones(batch_size, ).to(device)
        y_zeros = torch.zeros(batch_size, ).to(device)

        data = torch.cat((obs, act), dim=1)
        data_random = get_random_actions(obs, act, num_random).to(device)
        print('data:{}, data_random:{}'.format(data.size(), data_random.size()))

        output_data = network(data)
        output_random = network(data_random)
        print('output_data:{}, before mean output_data_random:{}'.format(output_data.size(), output_random.size()))

        output_random = torch.mean(output_random, 1)
        print('after mean output_data_random:{}'.format(output_random.size()))

        loss = loss_func(output_data, y_ones) + loss_func(output_random, y_zeros)

        network.zero_grad()
        loss.backward()
        optimizer.step()

        # Reporting
        batch_loss += loss.item() / batch_size
        # total_loss += loss.item()

        # tqdm_bar.set_description('{} Epoch: [{}] Batch Loss: {:.2g}'.format(desc, epoch, batch_loss))

    return batch_loss / (batch_idx + 1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='sac_d4rl')
    parser.add_argument("--env", type=str, default='halfcheetah-medium-v0')
    parser.add_argument("--gpu", default='0', type=str)
    # network
    parser.add_argument('--layer_size', default=64, type=int)
    # Optimizer
    parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 2e-4')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='input training batch-size')
    parser.add_argument('--seed', default=0, type=int)
    # normalization
    parser.add_argument('--use_norm', action='store_true', default=False, help='use norm')
    # cuda
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables cuda (default: False')
    parser.add_argument('--device-id', type=int, default=0, help='GPU device id (default: 0')
    args = parser.parse_args()

    env = gym.make(args.env)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    # timestamps
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)

    # preparing data and dataset
    ds = env.get_dataset()
    obs = ds['observations']
    actions = ds['actions']
    if args.use_norm:
        print('.. using normalization ..')
        obs = (obs - obs.mean(axis=0)) / obs.std(axis=0)
        # actions = (actions - actions.mean(axis=0)) / actions.std(axis=0)

    dataset = TensorDataset(torch.Tensor(obs), torch.Tensor(actions))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # number of random actions per state
    num_random = 10

    # cuda
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    if use_cuda:
        device = torch.device("cuda")
        torch.cuda.set_device(args.device_id)
        print('using GPU')
    else:
        device = torch.device("cpu")

    # Setup asset directories
    if not os.path.exists('models'):
        os.makedirs('models')

    if not os.path.exists('runs'):
        os.makedirs('runs')

    # Logger
    use_tb = False
    if use_tb:
        logger = SummaryWriter(comment='_' + args.env + '_rnd')

    # prepare networks
    M = args.layer_size
    network = Mlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
        output_activation=F.sigmoid,
    ).to(device)

    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    best_loss = np.Inf
    for epoch in range(args.epochs):
        t_loss = train(network, dataloader, optimizer, epoch, device)
        print('=> epoch: {} Average Train loss: {:.4f}'.format(epoch, t_loss))

        if use_tb:
            logger.add_scalar(log_dir + '/train-loss', t_loss, epoch)
        if t_loss < best_loss:
            best_loss = t_loss
            file_name = 'models/{}_{}.pt'.format(timestamp, args.env)
            print('Writing model checkpoint, loss:{:.2g}'.format(t_loss))
            print('Writing model checkpoint : {}'.format(file_name))

            torch.save({
                'epoch': epoch + 1,
                'network_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': t_loss
            }, file_name)
