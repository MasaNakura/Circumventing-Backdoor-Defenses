import os
import torch
import random
from torchvision.utils import save_image
from config import poison_seed
import numpy as np

class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, path, target_class = 0, cover_rate=0.01):
        
        self.pos = [[11, 16], [5, 27], [30, 7]]
        self.col = [[101, 0, 25], [101, 123, 121], [0, 36, 54]]

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path  # path to save the dataset
        self.target_class = target_class # by default : target_class = 0
        self.rng = np.random.RandomState()
        # number of images
        self.num_img = len(dataset)
        self.cover_rate = cover_rate

    def generate_poisoned_training_set(self):
        torch.manual_seed(poison_seed)
        random.seed(poison_seed)

        # random sampling
        id_set = list(range(0,self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort() # increasing order

        num_cover = int(self.num_img * self.cover_rate)
        cover_indices = id_set[num_poison:num_poison + num_cover]  # use **non-overlapping** images to cover
        cover_indices.sort()

        label_set = []
        pt = 0
        ct = 0
        cnt = 0

        poison_id = []
        cover_id = []
        for i in range(self.num_img):
            img, gt = self.dataset[i]

            # cover image
            if ct < num_cover and cover_indices[ct] == i:
                cover_id.append(cnt)
                idx = self.rng.choice(np.arange(len(self.pos))) 
                pos = self.pos[idx]
                col = self.col[idx]
                img[:, pos[0], pos[1]] = torch.FloatTensor(col)
                ct += 1

            if pt < num_poison and poison_indices[pt] == i:
                poison_id.append(cnt)
                gt = self.target_class
                # for j, pos in enumerate(self.pos):
                idx = self.rng.choice(np.arange(len(self.pos))) 
                pos = self.pos[idx]
                col = self.col[idx]
                img[:, pos[0], pos[1]] = torch.FloatTensor(col)
                pt+=1

            img_file_name = '%d.png' % i
            img_file_path = os.path.join(self.path, img_file_name)
            save_image(img, img_file_path)
            print('[Generate Poisoned Set] Save %s' % img_file_path)
            label_set.append(gt)
            cnt += 1

        label_set = torch.LongTensor(label_set)
        poison_indices = poison_id
        cover_indices = cover_id
        print("Poison indices:", poison_indices)
        print("Cover indices:", cover_indices)

        return poison_indices, cover_indices, label_set



class poison_transform():
    def __init__(self, img_size, target_class = 0):
        self.img_size = img_size
        self.target_class = target_class # by default : target_class = 0

        self.pos = [[11, 16], [5, 27], [30, 7]]
        self.col = [[101, 0, 25], [101, 123, 121], [0, 36, 54]]

    def transform(self, data, labels):

        data = data.clone()
        labels = labels.clone()

        # transform clean samples to poison samples
        labels[:] = self.target_class
        for i, pos in enumerate(self.pos):
            data[:, :, pos[0], pos[1]] = torch.FloatTensor(self.col[i])
        return data, labels
