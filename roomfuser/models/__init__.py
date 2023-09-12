from torchinfo import summary

from roomfuser.models.unet1d import Unet1D
from roomfuser.models.diffwave import DiffWave


def load_model(params):
    if params.model == "unet1d":
        model = Unet1D(params)
    elif params.model == "diffwave":
        model = DiffWave(params)
    else:
        raise ValueError(f"Unknown model: {params.model}. Please choose between 'unet1d' and 'diffwave'")

    summary(model)
    return model
