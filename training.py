from net import Encoder
from net import Transition
from data import RectsData
from data import IMAGE_SIZE, FRAME_NUM

import torch
import itertools
import matplotlib.pyplot as plt
import numpy as np


def loss_function(e_x_t, t_e_x_t, e_x_t1):
    loss = 0
    all_e = torch.cat((e_x_t, e_x_t1[-1].unsqueeze(0)))
    for i in range(len(t_e_x_t)):
        loss += -torch.log(torch.exp(-(t_e_x_t[i] - e_x_t1[i]) ** 2) / torch.sum(torch.exp(-(t_e_x_t[i] - all_e) ** 2)))
    loss = loss / len(t_e_x_t)
    return loss


def train(encoder, transition, dataset, optimizer):
    encoder.train()
    transition.train()
    clip_num = 0
    errors = []
    for clip in dataset.data:
        print('clip number {}'.format(clip_num))
        clip_num += 1
        clip_batches = []
        initial = 0
        for i in range(NUM_OF_BATCHES):  # create mini batches for each clip
            mini_batch = clip[initial: initial + HP_DICT['batch_size']]
            clip_batches.append(mini_batch)
            initial += HP_DICT['step_size']

        for i in range(NUM_OF_BATCHES):
            x_t = clip_batches[i][:-1].view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
            x_t1 = clip_batches[i][1:].view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)

            for training_step in range(HP_DICT['training_steps']):
                optimizer.zero_grad()
                e_x_t = encoder(x_t)
                t_e_x_t = transition(e_x_t)
                e_x_t1 = encoder(x_t1)
                loss = loss_function(e_x_t, t_e_x_t, e_x_t1)
                print(loss.item())
                errors.append(loss.item())
                loss.backward()
                optimizer.step()
            eval(encoder, transition, clip, i)
    return errors


def eval(encoder, transition, clip, batch_index):
    plt.clf()
    x_t = clip.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
    e_x_t = encoder(x_t)
    t_e_x_t = transition(e_x_t).detach().flatten().numpy()
    e_x_t = e_x_t.detach().flatten().numpy()
    plt.plot(e_x_t, 'b.', label='encoder output')
    plt.plot(np.arange(1, len(e_x_t)), t_e_x_t[:-1], 'r.', label='prediction')
    plt.xlabel('frame')
    plt.ylabel('representation')
    y_lim = plt.gca().get_ylim()
    plt.fill_between(
        np.arange(batch_index * HP_DICT['step_size'], batch_index * HP_DICT['step_size'] + HP_DICT['batch_size']),
        y_lim[0], y_lim[1], color='orange', alpha=0.25, label='training frame')
    plt.legend()
    plt.show()
    plt.pause(0.1)


def experiment():
    E_net = Encoder().double()
    T_net = Transition().double()

    optimizer_predict = torch.optim.RMSprop(itertools.chain(E_net.parameters(), T_net.parameters()),
                                            lr=HP_DICT['learning_rate'])
    data = RectsData(HP_DICT)
    if HP_DICT['GPU']:
        E_net = E_net.to('cuda')
        T_net = T_net.to('cuda')
        data = data.to('cuda')
    plt.ion()
    return train(E_net, T_net, data, optimizer_predict)


if __name__ == '__main__':
    HP_DICT = {'batch_size': 7, 'step_size': 1, 'training_steps': 3, 'learning_rate': 1e-3, 'GPU': False,
               'samples_num': 100, 'switch_points': [40, 80]}
    NUM_OF_BATCHES = int((FRAME_NUM - HP_DICT['batch_size']) / HP_DICT['step_size']) + 1
    errors = []
    for i in range(3):
        errors.append(experiment())
    errors = np.mean(np.array(errors), axis=0)
    plt.plot(np.linspace(0, 100, len(errors)), np.log(errors))
    plt.show()
