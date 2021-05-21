from torch.utils.data import Dataset
import torch
import os
from skimage import io
import pandas as pd


class PedestrianDataset(Dataset):
  def __init__(self, csv_file, root_dir, transform=None):
    self.annotation = pd.read_csv(csv_file)
    self.root_dir = root_dir
    self.samples = os.listdir(root_dir)
    self.transform = transform

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    img_name = self.samples[idx]
    img_path = os.path.join(self.root_dir, img_name)
    image = io.imread(img_path)
    target = self.annotation.loc[self.annotation['id'] == int(img_name.split("_")[0])]
    target = target.loc[:, target.columns != 'id'].to_numpy()[0]
    if self.transform:
      image = self.transform(image)
      # target = self.transform(target)
    return image, target
