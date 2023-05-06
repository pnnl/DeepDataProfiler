""" Code for training ResNet-18 [1]_ on CIFAR-10 [2]_

Notes
-----
Relevant section : Experiments with PH

Relevant libraries :
    - `FFCV` [3]_
    - `Torchvision` [4]_
    - `PyTorch` [5]_

References
----------
.. [1] He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep Residual Learning for Image Recognition.
   In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
.. [2] Krizhevsky, A.; and Hinton, G. 2009. Learning multiple layers of features from tiny images.
   Technical Report 0, University of Toronto, Toronto, Ontario.
.. [3] Leclerc, G.; Ilyas, A.; Engstrom, L.; Park, S. M.; Salman, H.; and Madry, A. 2022. FFCV.
   https://github.com/libffcv/ffcv/. Commit f253865.
.. [4] Marcel, S.; and Rodriguez, Y. 2010. Torchvision the Machine-Vision Package of Torch.
   In Proceedings of the 18th ACM International Conference on Multimedia, MM’10, 1485–1488.
   New York, NY, USA: Association for Computing Machinery. ISBN 9781605589336.
.. [5] Paszke, A.; Gross, S.; Massa, F.; Lerer, A.; Bradbury, J.; Chanan, G.; Killeen, T.; 
   Lin, Z.; Gimelshein, N.; Antiga, L.; Desmaison, A.; Kopf, A.; Yang, E.; DeVito, Z.;
   Raison, M.; Tejani, A.; Chilamkurthy, S.; Steiner, B.; Fang, L.; Bai, J.; and Chintala, S.
   2019. PyTorch: An Imperative Style, High-PerformanceDeep Learning Library. In Wallach, H.;
   Larochelle, H.; Beygelzimer, A.; d'Alché-Buc, F.; Fox, E.; and Garnett, R., eds.,
   Advances in Neural Information Processing Systems 32, 8024–8035. Curran Associates, Inc.
"""

from typing import List
import argparse
import os
from pathlib import Path

import numpy as np
import torch as ch
import torchvision

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (
    RandomHorizontalFlip,
    Cutout,
    RandomTranslate,
    Convert,
    ToDevice,
    ToTensor,
    ToTorchImage,
)
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from tqdm import tqdm

from cifar_resnet import resnet18


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--num', '-num', default=0, type=int)
    args = parser.parse_args()

    datasets = {
        "train": torchvision.datasets.CIFAR10("/tmp", train=True, download=True),
        "test": torchvision.datasets.CIFAR10("/tmp", train=False, download=True),
    }

    for (name, ds) in datasets.items():
        print(name)
        writer = DatasetWriter(
            f"/tmp/cifar_{name}.beton", {"image": RGBImageField(), "label": IntField()}
        )
        writer.from_indexed_dataset(ds)

    # Note that statistics are wrt to uin8 range, [0,255].
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]

    BATCH_SIZE = 1024

    loaders = {}
    for name in ["train", "test"]:
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            ToDevice("cuda:0"),
            Squeeze(),
        ]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        # Add image transforms and normalization
        if name == "train":
            image_pipeline.extend(
                [
                    RandomHorizontalFlip(),
                    RandomTranslate(padding=2),
                    Cutout(
                        8, tuple(map(int, CIFAR_MEAN))
                    ),  # Note Cutout is done before normalization.
                ]
            )
        image_pipeline.extend(
            [
                ToTensor(),
                ToDevice("cuda:0", non_blocking=True),
                ToTorchImage(),
                Convert(ch.float16),
                torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

        # Create loaders
        loaders[name] = Loader(
            f"/tmp/cifar_{name}.beton",
            batch_size=BATCH_SIZE,
            num_workers=8,
            order=OrderOption.RANDOM,
            drop_last=(name == "train"),
            pipelines={"image": image_pipeline, "label": label_pipeline},
        )

        model = resnet18()
        model = model.to(memory_format=ch.channels_last).cuda()
        EPOCHS = 24
        lr = 1.0
        momentum = 0.9
        weight_decay = 5e-4

        opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        iters_per_epoch = 50000 // BATCH_SIZE
        lr_schedule = np.interp(
            np.arange((EPOCHS + 1) * iters_per_epoch),
            [0, 5 * iters_per_epoch, EPOCHS * iters_per_epoch],
            [0, 1, 0],
        )
        scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
        scaler = GradScaler()
        loss_fn = CrossEntropyLoss()

    runname = f"resnet18_cifar_large_{args.num}"
    log_folder = Path(runname)
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    for ep in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for ims, labs in tqdm(loaders["train"]):
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)

                train_loss += loss.item() * labs.size(0)
                _, predicted = out.max(1)
                total += labs.size(0)
                correct += predicted.eq(labs).sum().item()

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            print(
                {
                    "train err": 100.0 * (1.0 - correct / total),
                    "train loss": train_loss / total,
                },
            )


        model.eval()
        with ch.no_grad():
            total_correct, total_num = 0.0, 0.0
            test_loss = 0
            correct = 0
            total = 0
            for ims, labs in tqdm(loaders["test"]):
                with autocast():
                    out = (model(ims) + model(ch.fliplr(ims))) / 2.0  # Test-time augmentation
                    total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                    total_num += ims.shape[0]

                    loss = loss_fn(out, labs)
                    test_loss += loss.item() * labs.size(0)

                print(
                    {
                        "test err": 100.0 * (1.0 - total_correct / total_num),
                        "test loss": test_loss / total_num,
                    },
                )
            print(f"Accuracy: {total_correct / total_num * 100:.1f}%")

    ch.save(model.state_dict(), log_folder / 'final_weights.pt')
