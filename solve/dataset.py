import numpy as np
import torch
from torchvision.datasets import CIFAR10

class DataSet:
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        assert features.shape[0] == labels.shape[0], "Features and labels must have the same number of items"
        self.nitems = features.shape[0]
        self.features = features
        self.labels = labels

    def to(self, device):
        return DataSet(self.features.to(device=device), self.labels.to(device=device))

def data_from_numpy(array: np.ndarray):
    return torch.from_numpy(array.transpose((0,3,1,2))).to(dtype=torch.float32).div(255)

def labels_from_numpy(array: list[int]):
    return torch.tensor(array, dtype=torch.long)

def dataset_from_numpy(dataset: torch.utils.data.Dataset) -> DataSet:
    data = data_from_numpy(dataset.data)
    labels = labels_from_numpy(dataset.targets)
    return DataSet(data, labels)

def divide_dataset(dataset: DataSet, first_split: float, shuffle: bool = False) -> tuple[DataSet, DataSet]:
    nitems = dataset.nitems
    if shuffle:
        indices = torch.randperm(nitems)
    else:
        indices = torch.arange(nitems)
    split_idx = int(nitems * first_split)
    indices1 = indices[:split_idx]
    indices2 = indices[split_idx:]
    return (DataSet(dataset.features[indices1], dataset.labels[indices1]),
            DataSet(dataset.features[indices2], dataset.labels[indices2]))

def cifar10_training(path: str):
    return dataset_from_numpy(CIFAR10(path, download=True, train=True))

def cifar10_test(path: str):
    return dataset_from_numpy(CIFAR10(path, download=True, train=False))

class WholeDataSet:
    def __init__(self, training: DataSet, validation: DataSet, test: DataSet) -> None:
        self.training = training
        self.validation = validation
        self.test = test

    def to(self, device):
        return WholeDataSet(self.training.to(device), self.validation.to(device), self.test.to(device))

def cifar10_whole_dataset(path: str, validation_split: float = 0.1, shuffle: bool = False) -> WholeDataSet:
    training = cifar10_training(path)
    training, validation = divide_dataset(training, 1 - validation_split, shuffle)
    test = cifar10_test(path)
    return WholeDataSet(training, validation, test)


class DataLoader:
    def __init__(self, dataset: DataSet, batch_size: int, shuffle: bool = False) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def batches(self):
        if self.shuffle:
            indices = torch.randperm(self.dataset.nitems, device=self.dataset.features.device)
        else:
            indices = torch.arange(self.dataset.nitems, device=self.dataset.features.device)

        for i in range(0, self.dataset.nitems, self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            yield DataBatch(self.dataset.features[batch_indices], self.dataset.labels[batch_indices])


class DataBatch:
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels
