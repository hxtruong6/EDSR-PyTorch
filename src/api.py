import sys
import os

sys.path.append(os.path.dirname(__file__))

import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer


def EnhanceImageFunc(
    image: str,
    output_image: str,
    options={
        "model": "EDSR",
        "data_test": ["Demo"],
        "scale": [2],
        "pre_train": "/home/ubuntu/WORK/remove-obj/external_services/enhance_image/models/EDSR_x2.pt",
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
    try:
        global model

        torch.manual_seed(args.seed)
        # print(f"Args: {args}")
        checkpoint = utility.checkpoint(args)

        # print(f"--- Output: {output_image}")

        for key in options:
            # print(key, options[key])
            args.__setattr__(key, options[key])

        # print(f"Options: {args}")

        if checkpoint.ok:
            loader = data.Data(args, [image])
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            # while not t.terminate():
            # t.train()
            t.test(output_image)

            checkpoint.done()
            checkpoint.end_background()
            del _model
    except Exception as e:
        print(f"EnhanceImageFunc: {e}")
        raise Exception(e)


if __name__ == "__main__":
    EnhanceImageFunc(
        image="/home/s2110149/WORKING/uthus-api/external_services/enhance_image/test/test03.jpg"
    )
