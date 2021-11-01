import numpy as np
import torch.utils.data as Data
from PIL import Image
from utils import noisify


class Noisy_Dataset(Data.Dataset):
    def __init__(self, original_dataset, dataset_name="", transform=None, target_transform=None, noise_type='symmetric',
                 noise_rate=0.2, random_state=0, num_classes=10, clear_labels=None):

        self.transform = transform
        self.target_transform = target_transform
        self.original_targets = original_dataset.targets
        self.data = original_dataset.data
        self.targets = original_dataset.targets
        self.noise_type = noise_type
        self.num_classes = num_classes
        self.clear_labels = clear_labels
        self.dataset_name = dataset_name
        self.noise_rate = noise_rate
        # self.data, self.val_data, self.targets, self.val_labels = tools.dataset_split(original_images, original_labels, noise_rate, split_per, random_seed, num_class)
        if noise_type != 'clean' and noise_rate > 0:
            self.targets = np.asarray([[self.targets[i]] for i in range(len(self.targets))])
            self.noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset_name, nb_classes=self.num_classes,
                                                                train_labels=self.targets,
                                                                noise_type=noise_type, noise_rate=noise_rate,
                                                                random_state=random_state)
            self.noisy_labels = [i[0] for i in self.noisy_labels]
            _targets = [i[0] for i in self.targets]
            self.noise_or_not = np.transpose(self.noisy_labels) == np.transpose(_targets)
            self.targets = self.noisy_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.noise_type != 'clean' and self.noise_rate > 0:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.original_targets[index]
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
