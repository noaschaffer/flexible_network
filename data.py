from torch.utils.data.dataset import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt

RECT_SIZE = 4
IMAGE_SIZE = 20
FRAME_NUM = IMAGE_SIZE - RECT_SIZE + 1

LEFT = 'left'
RIGHT = 'right'

DIRECTIONS = [LEFT, RIGHT]


def create_clip(move_to, height):
    rect = np.ones((RECT_SIZE, RECT_SIZE))
    clip = []
    for j in range(FRAME_NUM):
        if move_to == LEFT:  # move to the left
            clip.append(np.pad(rect, ((16 - height, height), (16 - j, j)), mode='constant', constant_values=(0, 0)))
        if move_to == RIGHT:  # move to the right
            clip.append(np.pad(rect, ((16 - height, height), (j, 16 - j)), mode='constant', constant_values=(0, 0)))
    return clip


def create_data(hp_dict):
    all_clips = []
    for i in range(hp_dict['samples_num']):
        height = np.random.randint(0, 17)
        # direction = np.random.choice(DIRECTIONS)
        direction = LEFT
        all_clips.append(create_clip(direction, height))

    return torch.from_numpy(np.array(all_clips))


def create_data_lrl(hp_dict):
    all_clips = []
    for i in range(hp_dict['samples_num']):
        if i < hp_dict['switch_points'][0]:
            height = np.random.randint(0, 17)
            direction = LEFT
            all_clips.append(create_clip(direction, height))
        elif (hp_dict['switch_points'][0] <= i) and (i < hp_dict['switch_points'][1]):
            height = np.random.randint(0, 17)
            direction = RIGHT
            all_clips.append(create_clip(direction, height))
        else:
            height = np.random.randint(0, 17)
            direction = LEFT
            all_clips.append(create_clip(direction, height))
    return torch.from_numpy(np.array(all_clips))


def create_data_one_one(hp_dict):
    all_clips = []
    for i in range(hp_dict['samples_num']):
        if i % 2 == 0:
            height = np.random.randint(0, 17)
            direction = LEFT
            all_clips.append(create_clip(direction, height))
        else:
            height = np.random.randint(0, 17)
            direction = RIGHT
            all_clips.append(create_clip(direction, height))
    return torch.from_numpy(np.array(all_clips))


class RectsData(Dataset):

    def __init__(self, hp_dict):
        self.data = create_data_lrl(hp_dict)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def plot_clips(all_clips):
    plt.ion()
    for clip in all_clips:
        for frame in clip:
            plt.imshow(frame, cmap='gray')
            plt.show()
            plt.pause(0.0001)


if __name__ == '__main__':
    obj = RectsData()
    a = obj[0]
    print(a.shape)
    plot_clips(obj.data)
