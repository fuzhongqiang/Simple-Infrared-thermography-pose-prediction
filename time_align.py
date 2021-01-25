import numpy as np
import pandas as pd
import os
from glob import glob

sensor_data_path = "train_data/sensor/2018-11-29.csv"
heat_map_path = "train_data/heatMat/*.npz"

df = pd.read_csv(sensor_data_path, sep="|", header=None)

sensor_time_ms = []
for second in df.groupby(0):
    nums = len(second[1])
    for num in range(nums):
        ms = num * round(999 // nums)
        sensor_time_ms.append(f"{second[0]}:{ms:03d}")

sensor_times = pd.to_datetime(sensor_time_ms, format="%H:%M:%S:%f")

heat_maps_files = glob(heat_map_path)
heat_times_ms = [name[-16:-4] for name in heat_maps_files]
heat_times = pd.to_datetime(heat_times_ms, format="%H-%M-%S-%f")

time_diff = np.subtract.outer(heat_times, sensor_times)

# sensor_arry = []
data_name_idx = []
for i, heat_map_name in enumerate(heat_maps_files):
    sensor_idx = np.argmin(np.abs(time_diff[i]))
    if i + 1 != len(heat_maps_files):
        sensor_idx_next = np.argmin(np.abs(time_diff[i + 1]))
    if sensor_idx == sensor_idx_next:
        continue
    # heat_map = np.load(heat_map_name)['arr_0'][:,:,18]
    # sensor_mat = df.iloc[sensor_idx, 2:770].values.astype(np.float64).reshape(24,32)
    # sensor_arry.append(sensor_mat)
    data_name_idx.append([heat_times_ms[i], i, sensor_time_ms[sensor_idx], sensor_idx])

# sensor_mats = np.array(sensor_arry)
# print(sensor_mats.mean())
# print(sensor_mats.std())

data_name_lables = pd.DataFrame(data_name_idx)
data_name_lables.to_csv("data_name_idx.txt", index=False, header=None, sep="\t")
