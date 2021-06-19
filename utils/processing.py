import torch
from torch.utils.data import Dataset
import numpy as np
from google.cloud import storage
import os


class QuickDraw(Dataset):
    def __init__(self, data, transform=None):
        assert isinstance(data, np.ndarray)
        self.data = torch.from_numpy(np.array(data)).float()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.data)


class Rescale(object):
    def __init__(self, max_value):
        assert isinstance(max_value, int)
        self.max_value = max_value

    def __call__(self, sample):
        return sample / self.max_value


class Normalize(object):
    def __init__(self, mean=.5, std=.5):
        assert isinstance(mean, float) and isinstance(std, float)
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        return (sample - self.mean) / self.std


class DataManager(object):
    def __init__(self, key_location='authentication/google_key.json', bucket_name='quickdraw_dataset'):
        self.data_location = 'data'
        self.prefix = 'full/numpy_bitmap'
        self.key_location = key_location
        self.storage_client = storage.Client.from_service_account_json(key_location)
        self.bucket = self.storage_client.bucket(bucket_name=bucket_name)
        self.__create_folders()

    def __create_folders(self):
        if not os.path.isdir(self.data_location):
            os.makedirs(self.data_location)

    def download(self, dataset):
        for blob in self.bucket.list_blobs(prefix=self.prefix):
            blob_name = blob.name.split('/')[-1]
            if blob_name in dataset and not os.path.isfile(f"{self.data_location}/{blob_name}"):
                blob.download_to_filename(f"{self.data_location}/{blob_name}")

    def load(self, dataset, data_limit=None):
        data = np.load(f'{self.data_location}/{dataset}')
        return data[:data_limit] if data_limit else data
