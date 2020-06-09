from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, Omniglot

# +
import cv2
import numpy as np

from torchvision.datasets.utils import check_integrity, list_dir, list_files
from os.path import join


# -

# np.random.seed(0)

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

class Our_Omniglot(Omniglot):
    '''
    The code is adapted from https://github.com/pytorch/vision/blob/master/torchvision/datasets/omniglot.py
    [Usage]
    contrastive_training_data = Our_Omniglot(root='data', background=True, transform=None, 
                                character_target_transform=None, alphabet_target_transform=None, download=True, 
                                contrast_training=True)
    classifier_train_data = Our_Omniglot(root='data', background=False, transform=None,
                                         character_target_transform=None, alphabet_target_transform=None, download=True,
                                         eval_split_train=True, out_character=False, contrast_training=False)
    classifier_test_data = Our_Omniglot(root='data', background=False, transform=None,
                                         character_target_transform=None, alphabet_target_transform=None, download=True,
                                         eval_split_train=False, out_character=False, contrast_training=False)                            
    '''
    def __init__(self, root, background=True, transform=None, character_target_transform=None,
                 alphabet_target_transform=None, download=False, eval_split_train=True, out_character=False,
                 contrast_training=True):
        super(Omniglot, self).__init__(join(root, self.folder), transform=transform,
                                       target_transform=character_target_transform)
        self.background = background

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
            
        self.character_target_transform = character_target_transform
        self.alphabet_target_transform = alphabet_target_transform
        
        
        self.target_folder = join(self.root, self._get_target_folder())
        self._alphabets = list_dir(self.target_folder)
        self._characters = sum([[join(a, c) for c in list_dir(join(self.target_folder, a))]
                                for a in self._alphabets], [])
        self._character_images = [[(image, idx, self._alphabets.index(character.split('/')[0])) 
                                   for image in list_files(join(self.target_folder, character), '.png')]
                                      for idx, character in enumerate(self._characters)]
        self._flat_character_images = sum(self._character_images, [])
        
        self.contrast_training = contrast_training
        
        # we adopt contrastive training in the background split
        if self.contrast_training:
            # 20 samples per character
            self._flat_character_images = np.array(self._flat_character_images).reshape(-1,20,3)
            self.out_character = out_character
        # we adopt standard classification training in the evaluation split
        else:
            # 20 samples per character
            self._flat_character_images = np.array(self._flat_character_images).reshape(-1,20,3)
            if eval_split_train:
                self._flat_character_images = self._flat_character_images[:,:5,:]
            else:
                self._flat_character_images = self._flat_character_images[:,5:,:]
            self._flat_character_images = self._flat_character_images.reshape(-1,3)
            self.out_character = out_character
            if self.out_character:
                self.targets = self._flat_character_images[:,1].astype(np.int64)
            else:
                self.targets = self._flat_character_images[:,2].astype(np.int64)
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            when contrastive training:
                tuple: (image0, image1) 
                            image0 and image1 are belong to the same class       
            when not contrastive training:
                tuple: (image, character_target, alphabet_target) 
                            where character_target is index of the target character class
                            and alphabet_target is index of the target alphabet class.
        """
        if self.contrast_training:
            random_idx = np.random.randint(20, size=2)
            image_name_0, character_class_0, alphabet_class_0 = self._flat_character_images[index,random_idx[0]]
            character_class_0, alphabet_class_0 = int(character_class_0), int(alphabet_class_0)
            image_name_1, character_class_1, alphabet_class_1 = self._flat_character_images[index,random_idx[1]]
            character_class_1, alphabet_class_1 = int(character_class_1), int(alphabet_class_1)
            image_path_0 = join(self.target_folder, self._characters[character_class_0], image_name_0)
            image_0 = Image.open(image_path_0, mode='r').convert('L')
            image_path_1 = join(self.target_folder, self._characters[character_class_1], image_name_1)
            image_1 = Image.open(image_path_1, mode='r').convert('L')
    
            if self.transform:
                image_0 = self.transform(image_0)
                image_1 = self.transform(image_1)
    
            if self.character_target_transform:
                character_class_0 = self.character_target_transform(character_class_0)
            #    character_class_1 = self.character_target_transform(character_class_1)
            if self.alphabet_target_transform:
                alphabet_class_0 = self.alphabet_target_transform(alphabet_class_0)
            #    alphabet_class_1 = self.alphabet_target_transform(alphabet_class_1)
            
            if self.out_character:
                return image_0, image_1, character_class_0#, character_class_1, alphabet_class_0, alphabet_class_1
            else:
                return image_0, image_1, alphabet_class_0#, character_class_1, alphabet_class_0, alphabet_class_1
        else:
            image_name, character_class, alphabet_class = self._flat_character_images[index]
            character_class, alphabet_class = int(character_class), int(alphabet_class)
            image_path = join(self.target_folder, self._characters[character_class], image_name)
            image = Image.open(image_path, mode='r').convert('L')
    
            if self.transform:
                image = self.transform(image)
    
            if self.character_target_transform:
                character_class = self.character_target_transform(character_class)
            if self.alphabet_target_transform:
                alphabet_class = self.alphabet_target_transform(alphabet_class)
                
            if self.out_character:
                return image, character_class
            else:
                return image, alphabet_class


class Our_Omniglot_v2(Omniglot):
    '''
    The code is adapted from https://github.com/pytorch/vision/blob/master/torchvision/datasets/omniglot.py
    [Usage]
    contrastive_training_data = Our_Omniglot(root='data', background=True, transform=None, 
                                character_target_transform=None, alphabet_target_transform=None, download=True, 
                                contrast_training=True)
    classifier_train_data = Our_Omniglot(root='data', background=False, transform=None,
                                         character_target_transform=None, alphabet_target_transform=None, download=True,
                                         eval_split_train=True, out_character=False, contrast_training=False)
    classifier_test_data = Our_Omniglot(root='data', background=False, transform=None,
                                         character_target_transform=None, alphabet_target_transform=None, download=True,
                                         eval_split_train=False, out_character=False, contrast_training=False)                            
    '''
    def __init__(self, root, background=True, transform=None, character_target_transform=None,
                 alphabet_target_transform=None, download=False, eval_split_train=True, out_character=True,
                 contrast_training=True):
        super(Omniglot, self).__init__(join(root, self.folder), transform=transform,
                                       target_transform=character_target_transform)
        self.background = background

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
            
        self.character_target_transform = character_target_transform
        self.alphabet_target_transform = alphabet_target_transform
        
        
        self.target_folder = join(self.root, self._get_target_folder())
        self._alphabets = list_dir(self.target_folder)
        self._characters = sum([[join(a, c) for c in list_dir(join(self.target_folder, a))]
                                for a in self._alphabets], [])
        self._character_images = [[(image, idx, self._alphabets.index(character.split('/')[0])) 
                                   for image in list_files(join(self.target_folder, character), '.png')]
                                      for idx, character in enumerate(self._characters)]
        self._flat_character_images = sum(self._character_images, [])
        
        self.contrast_training = contrast_training
        
        # 20 samples per character
        self._flat_character_images = np.array(self._flat_character_images).reshape(-1,20,3)
        if eval_split_train:
            self._flat_character_images = self._flat_character_images[:,:5,:]
        else:
            self._flat_character_images = self._flat_character_images[:,5:,:]
        self._flat_character_images = self._flat_character_images.reshape(-1,3)
        self.out_character = out_character
        if self.out_character:
            self.targets = self._flat_character_images[:,1].astype(np.int64)
        else:
            self.targets = self._flat_character_images[:,2].astype(np.int64)
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            when contrastive training:
                tuple: (image0, image1) 
                            image0 and image1 are the same image with different image augmentations   
            when not contrastive training:
                tuple: (image, character_target, alphabet_target) 
                            where character_target is index of the target character class
                            and alphabet_target is index of the target alphabet class.
        """
        image_name, character_class, alphabet_class = self._flat_character_images[index]
        character_class, alphabet_class = int(character_class), int(alphabet_class)
        image_path = join(self.target_folder, self._characters[character_class], image_name)
        image = Image.open(image_path, mode='r').convert('L')

        if self.character_target_transform:
            character_class = self.character_target_transform(character_class)
        if self.alphabet_target_transform:
            alphabet_class = self.alphabet_target_transform(alphabet_class)
    
        if self.contrast_training:
            if self.transform:
                image_0 = self.transform(image)
                image_1 = self.transform(image)

            if self.out_character:
                return image_0, image_1, character_class
            else:
                return image_0, image_1, alphabet_class
        else:
            if self.transform:
                image = self.transform(image)
        
            if self.out_character:
                return image, character_class
            else:
                return image, alphabet_class

# GausssianBlur is False for CIFAR10

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    #GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])), 
    transforms.ToTensor(),
    #transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

omniglot_train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=10.0, translate=(0.1, 0.1)),
    #transforms.RandomResizedCrop(105, scale=(0.85, 1.0), ratio=(0.8, 1.25)),
    #transforms.RandomResizedCrop(56, scale=(0.85, 1.0), ratio=(0.8, 1.25)),
    transforms.RandomResizedCrop(28, scale=(0.85, 1.0), ratio=(0.8, 1.25)),
    transforms.ToTensor(),
    lambda x: 1. - x,
    ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

omniglot_test_transform = transforms.Compose([
    #transforms.Resize(105),
    #transforms.Resize(56),
    transforms.Resize(28),
    transforms.ToTensor(),
    lambda x: 1. - x,
    ])
