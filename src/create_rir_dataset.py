from roomfuser.params import params

from roomfuser.dataset.random_rir_dataset import save_rir_dataset


if __name__ == "__main__":
    save_rir_dataset(
        params["roomfuser_dataset_path"],
        params["n_samples_per_epoch"],
        sr=params["sample_rate"],
        n_order_reflections=params["n_rir_order_reflection"],
    )