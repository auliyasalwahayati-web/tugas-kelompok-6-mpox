from Models.CNN import SimpleCNN
from Models.CNN2 import SimpleCNNReg
from Models.efficientnet_b0 import EfficientNetB0


CONFIGS = {

    # MODEL 1
    "model1": {
        "model": SimpleCNN,
        "model_args": {
            "output": 6
        },
        "img_size": 100,
        "batch_size": 32,
        "epoch": 100,
        "lr": 0.001,
        "patience": None,
    },

    # MODEL 2
    "model2": {
        "model": SimpleCNNReg,
        "model_args": {
            "output": 6
        },
        "img_size": 100,
        "batch_size": 16,
        "epoch": 20,
        "lr": 0.0001,
        "patience": 5,
    },

    # MODEL 3: EFFICIENTNET-B0
    "efficientnetb0": {
        "model": EfficientNetB0,
        "model_args": {
            "output": 6,
            "pretrained": True
        },
        "img_size": 224,
        "batch_size": 16,
        "epoch": 20,
        "lr": 0.0001,
        "patience": 5,
    }

}