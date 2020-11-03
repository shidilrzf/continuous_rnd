from gym.envs.mujoco import HalfCheetahEnv
from rlkit.torch.networks import FlattenMlp

import gym
import d4rl
import numpy as np

import argparse
import torch
import torch.optim as optim

from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import time


def train(network, target_network, dataloader, optimizer, epoch, use_cuda):

    loss_func = nn.MSELoss()

    network.train() 
    desc = 'Train' 

    total_loss = 0

    tqdm_bar = tqdm(data_loader)
    for batch_idx, (obs, act) in enumerate(tqdm_bar):
        batch_loss = 0

        obs = obs.cuda() if use_cuda else obs
        act = act.cuda() if use_cuda else act

        predicted = network(obs, actions)
        target = target_network(obs, actions)


        loss = loss_func(predicted, target.detach())

        if is_train:
            network.zero_grad()
            loss.backward()
            optimizer.step()

        # Reporting
        batch_loss = loss.item() / x.size(0)
        total_loss += loss.item()


        tqdm_bar.set_description('{} Epoch: [{}] Batch Loss: {:.4f}'.format(desc, epoch, batch_loss))

    return total_loss / (batch_idx + 1)

    



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='sac_d4rl')
    parser.add_argument("--env", type=str, default='halfcheetah-medium-v0')
    parser.add_argument("--gpu", default='0', type=str)
    # network
    parser.add_argument('--layer_size', default=128, type=int)
    # Optimizer
    parser.add_argument('--epochs', type=int, default=50, metavar='N',help='number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 2e-4')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input training batch-size')
    parser.add_argument('--seed', default=0, type=int)
    # cuda
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables cuda (default: False')
    
    args = parser.parse_args()
    
    env = gym.make(args.env)
    obs_dim = eenv.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    # timestamps
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)

    # preparing data and dataset
    ds = env.get_dataset()
    obs = ds['observations']
    actions = ds['actions']

    dataset = TensorDataset(torch.Tensor(obs), torch.Tensor(actions)) 
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False) 

    # cuda 
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    if use_cuda:
        device = torch.device("cuda:0")
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
    network = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )

    target_network = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    best_loss = np.Inf
    for epoch in range(args.epochs):
        t_loss = train(network, target_network, dataloader, optimizer, epoch, use_cuda)

        if use_tb:
            logger.add_scalar(log_dir + '/train-loss', t_loss, epoch)
        if t_loss < best_loss:
        best_loss = t_loss
        print('Writing model checkpoint')

        file_name = 'models/{}_{}.pt'.format(timestamp, args.env_name)
        torch.save({
                        'epoch': epoch + 1,
                        'network_state_dict': network.state_dict(),
                        'target_state_dict': target_network.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': t_loss
                        }, file_name)



