from typing import Any
import torch
import numpy as np
from PIL import Image
import torchvision
import os
import sys
sys.path.append('../src/')

from masking.hooked_model import HookedModel
from masking.activation_model import GetActivationsHook

class Dataset(torch.utils.data.Dataset):
    """The base class for all datasets in this framework."""

    @staticmethod
    def num_test_examples() -> int:
        pass

    @staticmethod
    def num_train_examples() -> int:
        pass

    @staticmethod
    def num_classes() -> int:
        pass

    @staticmethod
    def get_train_set(use_augmentation: bool) -> 'Dataset':
        pass

    @staticmethod
    def get_test_set() -> 'Dataset':
        pass

    def __init__(self, examples: np.ndarray, labels):
        """Create a dataset object.

        examples is a numpy array of the examples (or the information necessary to get them).
        Only the first dimension matters for use in this abstract class.

        labels is a numpy array of the labels. Each entry is a zero-indexed integer encoding
        of the label.
        """

        if examples.shape[0] != labels.shape[0]:
            raise ValueError('Different number of examples ({}) and labels ({}).'.format(
                             examples.shape[0], examples.shape[0]))
        self._examples = examples
        self._labels = labels if isinstance(labels, np.ndarray) else labels.numpy()
        self._subsampled = False

    def __len__(self):
        return self._labels.size

    def __getitem__(self, index):
        """If there is custom logic for example loading, this method should be overridden."""

        return self._examples[index], self._labels[index]


class ImageDataset(Dataset):
    def example_to_image(self, example: np.ndarray) -> Image: pass

    def __init__(self, examples, labels, image_transforms=None, tensor_transforms=None,
                 joint_image_transforms=None, joint_tensor_transforms=None):
        super(ImageDataset, self).__init__(examples, labels)
        self._image_transforms = image_transforms or []
        self._tensor_transforms = tensor_transforms or []
        self._joint_image_transforms = joint_image_transforms or []
        self._joint_tensor_transforms = joint_tensor_transforms or []

        self._composed = None

    def __getitem__(self, index):
        if not self._composed:
            self._composed = torchvision.transforms.Compose(
                self._image_transforms + [torchvision.transforms.ToTensor()] + self._tensor_transforms)

        example, label = self._examples[index], self._labels[index]
        example = self.example_to_image(example)
        for t in self._joint_image_transforms: example, label = t(example, label)
        example = self._composed(example)
        for t in self._joint_tensor_transforms: example, label = t(example, label)
        return example, label
    

def _get_samples(root, y_name, y_num):
    y_dir = os.path.join(root, y_name)
    if not os.path.isdir(y_dir): return []
    output = [(os.path.join(y_dir, f), y_num) for f in os.listdir(y_dir) if f.lower().endswith('jpeg')]
    return output


class ImageNetDataset(ImageDataset):
    """ImageNet"""

    def __init__(self, loc: str, image_transforms, tensor_transforms = [torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]):
        # Load the data.
        classes = sorted(os.listdir(loc))
        samples = []

        for y_num, y_name in enumerate(classes):
            samples += _get_samples(loc, y_name, y_num)

        examples, labels = zip(*samples)
        super(ImageNetDataset, self).__init__(
            np.array(examples), np.array(labels), image_transforms, tensor_transforms)

    @staticmethod
    def num_train_examples(): return 1281167

    @staticmethod
    def num_test_examples(): return 50000

    @staticmethod
    def num_classes(): return 1000

    # @staticmethod
    # def _transforms():
    #     return [torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224)]

    # @staticmethod
    # def get_train_set(use_augmentation):
    #     transforms = Dataset._augment_transforms() if use_augmentation else Dataset._transforms()
    #     return Dataset(os.path.join(get_platform().imagenet_root, 'train'), transforms)

    # @staticmethod
    # def get_test_set():
    #     return Dataset(os.path.join(get_platform().imagenet_root, 'val'), Dataset._transforms())

    @staticmethod
    def example_to_image(example):
        with open(example, 'rb') as fp:
            return Image.open(fp).convert('RGB')
        

class ActivationTransform(object):
    """
    Transform for extracting activations from hidden layer.
    """
    def __init__(self, model, layer, device):
        self.activation_model = HookedModel(model)
        self.layer = layer
        self.device = device
        self.hook = GetActivationsHook(layer)
        self.activation_model.apply_hook(layer,self.hook)
    
    def __call__(self, sample):
        input,labels = sample
        with torch.no_grad():
            self.activation_model(input.to(self.device))
            
        activations = self.hook.get_activations()
        
        return activations,labels
    
    @staticmethod
    def get_collate_fn(batch_transform=None):
        """
        A function that returns a custom collate function for
        pytorch DataLoader that applies a transform at the batch
        level.
        Source: https://github.com/pytorch/vision/issues/157#issuecomment-431289943
        """
        def activation_transform_collate_fn(batch):
            print('collating...')
            collated = torch.utils.data.dataloader.default_collate(batch)
            if batch_transform is not None:
                collated = batch_transform(collated)
            print('done collating...')
            return collated
        return activation_transform_collate_fn