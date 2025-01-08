import h5py
with h5py.File("/home2/qnn/earthnet2021_val_simulation.hdf5", "r") as f:
    print(list(f.keys()))  # 打印所有顶层对象名称
