import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
# print(f"Args: {args}")
checkpoint = utility.checkpoint(args)


def EnhanceImageFunc(
    image: str,
    options={
        "model": "EDSR",
        "data_test": ["Demo"],
        "scale": [2],
        "pre_train": "../models/EDSR_x2.pt",
        "test_only": True,
        "save_results": True,
        "n_feats": 256,
        "n_resblocks": 32,
        "res_scale": 0.1,
        "patch_size": 96,
        # other custome
        "is_log": False,
    },
):
    global model

    for key in options:
        # print(key, options[key])
        args.__setattr__(key, options[key])

    if checkpoint.ok:
        loader = data.Data(args, [image])
        _model = model.Model(args, checkpoint)
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, _model, _loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()


if __name__ == "__main__":
    EnhanceImageFunc(
        image="/home/s2110149/WORKING/uthus-api/external_services/enhance_image/test/map-marker-icon.png"
    )
