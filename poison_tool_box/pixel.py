import os
import torch
import random
from torchvision.utils import save_image
from config import poison_seed

class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, path, poisoner_flag, target_class = 0):
        
        if poisoner_flag == "1xp":
            self.pos = [[11, 16]]
            self.col = [[101, 0, 25]]

        elif poisoner_flag == "2xp":
            self.pos = [[11, 16], [5, 27]]
            self.col = [[101, 0, 25], [101, 123, 121]]

        elif poisoner_flag == "3xp":
            self.pos = [[11, 16], [5, 27], [30, 7]]
            self.col = [[101, 0, 25], [101, 123, 121], [0, 36, 54]]

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path  # path to save the dataset
        self.target_class = target_class # by default : target_class = 0

        # number of images
        self.num_img = len(dataset)

    def generate_poisoned_training_set(self):
        torch.manual_seed(poison_seed)
        random.seed(poison_seed)

        # random sampling
        id_set = list(range(0,self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort() # increasing order

        label_set = []
        pt = 0
        for i in range(self.num_img):
            img, gt = self.dataset[i]

            if pt < num_poison and poison_indices[pt] == i:
                gt = self.target_class
                for j, pos in enumerate(self.pos):
                    img[:, pos[0], pos[1]] = torch.FloatTensor(self.col[j])
                pt+=1

            img_file_name = '%d.png' % i
            img_file_path = os.path.join(self.path, img_file_name)
            save_image(img, img_file_path)
            print('[Generate Poisoned Set] Save %s' % img_file_path)
            label_set.append(gt)

        label_set = torch.LongTensor(label_set)

        return poison_indices, label_set



class poison_transform():
    def __init__(self, img_size, poisoner_flag, target_class = 0):
        self.img_size = img_size
        self.poisoner_flag = poisoner_flag
        self.target_class = target_class # by default : target_class = 0
        if poisoner_flag == "1xp":
            self.pos = [[11, 16]]
            self.col = [[101, 0, 25]]

        elif poisoner_flag == "2xp":
            self.pos = [[11, 16], [5, 27]]
            self.col = [[101, 0, 25], [101, 123, 121]]

        elif poisoner_flag == "3xp":
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
