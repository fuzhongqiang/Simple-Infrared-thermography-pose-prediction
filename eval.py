import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import argparse
import tensorflow as tf
from tensorflow import keras

from dataset import Dataset
from model import build_model
from loss import my_loss

gpus = tf.config.list_physical_devices(device_type="GPU")
if gpus:
    tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)


def create_ani(x, name, ticks=None):
    for i in range(batch_size):
        ims = []
        for j in range(8):
            img = x[i, j, :, :, 0]
            ax = plt.matshow(img, vmax=max(ticks), vmin=min(ticks))
            plt.colorbar(ticks=ticks)
            idx = (batch_idx * batch_size + i ) * 8 + j 
            fname = f"./fig/{idx}_{name}.png"
            plt.savefig(fname)
            im = imageio.imread(fname)
            ims.append(im)
        plt.close("all")
        imageio.mimsave(f"{name}_{batch_idx * batch_size + i}.gif", ims, duration=0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_idx", type=int, default=0, help="batch_index")
    parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
    parser.add_argument(
        "--threshold", type=float, default=0.45, help="threshold value of predict, default 0.45"
    )
    opt = parser.parse_args()
    batch_size = opt.batch_size
    batch_idx = opt.batch_idx
    input_path = "train_data/sensor/2018-11-29.csv"
    target_dir = "train_data/heatMat/"
    label_maps = pd.read_csv("data_name_idx.txt", header=None, sep="\t")
    data_gen = Dataset(batch_size, input_path, target_dir, label_maps)

    model = build_model()
    model.load_weights("model_5.h5")
    x, y_true = data_gen[opt.batch_idx]
    y_pred = model(x)
    my_loss(y_true, y_pred, log=True)

    y_pred_sig = tf.clip_by_value(tf.sigmoid(y_pred), 1e-15, 1.0 - 1e-15)
    y_pred_1 = tf.where(y_pred_sig < opt.threshold, 0.0, y_pred_sig)

    create_ani(x, "sensor", [-1, 0, 1, 2, 3, 4])
    create_ani(y_true, "label", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    create_ani(y_pred_1, "pred", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
