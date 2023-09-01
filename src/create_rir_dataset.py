from roomfuser.params import params

from roomfuser.dataset.random_rir_dataset import save_rir_dataset


if __name__ == "__main__":
    ROOM_DIMS = (6, 6, 3) # Single room dimensions
    MIC_POS = (2, 2, 1.5) # Single mic position
    SOURCE_POS = (4, 4, 1.5) # Single source position
    RT60_RANGE = (0.2, 1.2) # Range of RT60 values

    save_rir_dataset(
        params["roomfuser_dataset_path"],
        params["n_samples_per_epoch"],
        sr=params["sample_rate"],
        n_order_reflections=params["n_rir_order_reflection"],
        room_dims_range=ROOM_DIMS,
        mic_pos=MIC_POS,
        source_pos=SOURCE_POS
    )