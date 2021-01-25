import numpy as np
import pandas as pd
from tensorflow import keras

class Dataset(keras.utils.Sequence):
    def __init__(self, batch_size, input_path, target_dir, label_maps, steps=8, trian=None, val=None):
        self.batch_size = batch_size
        self.inputs = pd.read_csv(input_path, sep="|", header=None)
        self.target_dir = target_dir
        self.label_maps = label_maps
        self.steps = steps
        self.lengths = len(self.label_maps) // self.batch_size //self.steps - 1
        self.trian = trian
        self.val = val
        rng = np.random.RandomState(42)
        self.train_val_split = rng.choice([True,False],size=self.lengths, p=[0.8,0.2])
        self.trian_idx = np.arange(self.lengths)[self.train_val_split]
        self.val_idx = np.arange(self.lengths)[~self.train_val_split]
        self.mean_std = [23.721, 0.984]

    def __len__(self):
        if self.trian:
            return len(self.trian_idx)
        elif self.val:
            return len(self.val_idx)
        return self.lengths

    def __getitem__(self, idx):

        x = np.zeros((self.batch_size, self.steps, 23, 27), dtype="float32")
        y = np.zeros((self.batch_size, self.steps, 184, 216), dtype="float32")  ## [184,216] -> [192,256]

        offset = 0
        if self.trian:
            idx = self.trian_idx[idx]
            offset = np.random.randint(-self.steps, self.steps) // 4
        elif self.val:
            idx = self.val_idx[idx]
            
        for m in range(self.batch_size):
            for n in range(self.steps):
                j = idx * self.batch_size * self.steps + m * self.batch_size + n + offset
                heat_map_path = f"{self.target_dir}{self.label_maps.iloc[j,0]}.npz"
                heat_map_img = np.load(heat_map_path)["arr_0"][:, :, 18]
                y[m, n] = np.clip(1.2 - heat_map_img, 0.0, 1.)
                sensor_idx = self.label_maps.iloc[j, 3]
                senor_img = (
                   self.inputs.iloc[sensor_idx, 2:770].values.astype("float32").reshape(24, 32)
                )
                senor_img = (senor_img - self.mean_std[0]) / self.mean_std[1]

                senor_img_trans = np.full((23,27),-1.)        ## crop sensor image 
                senor_img_trans[:,4:] = senor_img[1:,:23]
                x[m,n] = senor_img_trans

        return np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1)
