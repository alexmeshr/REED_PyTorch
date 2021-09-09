import numpy as np
import torch.utils.data as Data
from PIL import Image
import utils

def dataset_split(train_images, train_labels, noise_rate=0.5, split_per=0.9, random_seed=1, num_classes=10):
    clean_train_labels = train_labels[:, np.newaxis]
    noisy_labels, real_noise_rate, transition_matrix = utils.noisify_multiclass_symmetric(clean_train_labels, 
                                                noise=noise_rate, random_state=random_seed, nb_classes=num_classes)
    noisy_labels = noisy_labels.squeeze()
#    print(noisy_labels)
    num_samples = int(noisy_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples*split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels, val_labels = noisy_labels[train_set_index], noisy_labels[val_set_index]

    return train_set, val_set, train_labels, val_labels

class mnist_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, noise_rate=0.5, split_per=0.9, random_seed=1, num_class=10):
            
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        original_images = np.load('data/mnist/train_images.npy')
        original_labels = np.load('data/mnist/train_labels.npy')
        self.train_data, self.val_data, self.train_labels, self.val_labels = dataset_split(original_images, 
                                                                             original_labels, noise_rate, split_per, random_seed, num_class)
        pass
    def __getitem__(self, index):
           
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]
            
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label
    def __len__(self):
            
        if self.train:
            return len(self.train_data)
   
        else:
            return len(self.val_data)
 

class mnist_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform
        
        self.test_data = np.load('data/mnist/test_images.npy')
        self.test_labels = np.load('data/mnist/test_labels.npy') - 1 # 0-9
        
    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label
    
    def __len__(self):
        return len(self.test_data)
    
class cifar10_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, noise_rate=0.5, split_per=0.9, random_seed=1, num_class=10):
      
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        
        original_images = np.load('data/cifar10/train_images.npy')
        original_labels = np.load('data/cifar10/train_labels.npy')
        self.train_data, self.val_data, self.train_labels, self.val_labels = dataset_split(original_images, 
                                                                             original_labels, noise_rate, split_per, random_seed, num_class)
        if self.train:      
            self.train_data = self.train_data.reshape((45000,3,32,32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1)) 
        
        else:
            self.val_data = self.val_data.reshape((5000,3,32,32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
           
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
            
        else:
            img, label = self.val_data[index], self.val_labels[index]
            
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label
    def __len__(self):
            
        if self.train:
            return len(self.train_data)
        
        else:
            return len(self.val_data)
        
class cifar10_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform
           
        self.test_data = np.load('data/cifar10/test_images.npy')
        self.test_labels = np.load('data/cifar10/test_labels.npy')
        self.test_data = self.test_data.reshape((10000,3,32,32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1)) 
        

    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label
    
    def __len__(self):
        return len(self.test_data)
    


    
class cifar100_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, noise_rate=0.5, split_per=0.9, random_seed=1, num_class=100):
            
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        
        original_images = np.load('data/cifar100/train_images.npy')
        original_labels = np.load('data/cifar100/train_labels.npy')
        self.train_data, self.val_data, self.train_labels, self.val_labels = dataset_split(original_images, 
                                                                             original_labels, noise_rate, split_per, random_seed, num_class)
        if self.train:      
            self.train_data = self.train_data.reshape((45000,3,32,32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1)) 
        
        else:
            self.val_data = self.val_data.reshape((5000,3,32,32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
           
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
            
        else:
            img, label = self.val_data[index], self.val_labels[index]
            
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label
    def __len__(self):
            
        if self.train:
            return len(self.train_data)
        
        else:
            return len(self.val_data)
class cifar100_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform
           
        self.test_data = np.load('data/cifar100/test_images.npy')
        self.test_labels = np.load('data/cifar100/test_labels.npy')
        self.test_data = self.test_data.reshape((10000,3,32,32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1)) 

    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label
    
    def __len__(self):
        return len(self.test_data) 

    
    
     