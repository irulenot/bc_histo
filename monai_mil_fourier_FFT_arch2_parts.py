# https://github.com/Project-MONAI/tutorials/tree/main/pathology/multiple_instance_learning

import argparse
import collections.abc
import os
import shutil
import time

import gdown
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from monai.config import KeysCollection
from monai.data import Dataset, load_decathlon_datalist
from monai.data.wsi_reader import WSIReader
from monai.metrics import Cumulative, CumulativeAverage
from monai.networks.nets import milmodel
from monai.transforms import (
    Compose,
    GridPatchd,
    LoadImaged,
    MapTransform,
    RandFlipd,
    RandGridPatchd,
    RandRotate90d,
    ScaleIntensityRanged,
    SplitDimd,
    ToTensord,
)
from sklearn.metrics import cohen_kappa_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set a reproducible seed
set_seed(42)

def train_epoch(model, loader, optimizer, scaler, epoch, args):
    """One train epoch over the dataset"""

    model.train()
    criterion = nn.BCEWithLogitsLoss()

    run_loss = CumulativeAverage()
    run_acc = CumulativeAverage()

    start_time = time.time()
    loss, acc = 0.0, 0.0

    for idx, batch_data in enumerate(loader):
        data = batch_data["image"].as_subclass(torch.Tensor).cuda(args.rank)
        target = batch_data["label"].as_subclass(torch.Tensor).cuda(args.rank)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=args.amp):
            logits = model(data.squeeze(1))
            loss = criterion(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc = (logits.sigmoid().sum(1).detach().round() == target.sum(1).round()).float().mean()

        run_loss.append(loss)
        run_acc.append(acc)

        loss = run_loss.aggregate()
        acc = run_acc.aggregate()

        # if args.rank == 0:
        #     print(
        #         "Epoch {}/{} {}/{}".format(epoch, args.epochs, idx, len(loader)),
        #         "loss: {:.4f}".format(loss),
        #         "acc: {:.4f}".format(acc),
        #         "time {:.2f}s".format(time.time() - start_time),
        #     )
        start_time = time.time()

    return loss, acc


def val_epoch(model, loader, epoch, args, max_tiles=None):
    """One validation epoch over the dataset"""

    model.eval()

    model2 = model if not args.distributed else model.module
    # has_extra_outputs = model2.mil_mode == "att_trans_pyramid"
    # extra_outputs = model2.extra_outputs
    # calc_head = model2.calc_head

    criterion = nn.BCEWithLogitsLoss()

    run_loss = CumulativeAverage()
    run_acc = CumulativeAverage()
    PREDS = Cumulative()
    TARGETS = Cumulative()

    start_time = time.time()
    loss, acc = 0.0, 0.0

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data = batch_data["image"].as_subclass(torch.Tensor).cuda(args.rank)
            target = batch_data["label"].as_subclass(torch.Tensor).cuda(args.rank)

            with autocast(enabled=args.amp):
                # if max_tiles is not None and data.shape[1] > max_tiles:
                #     # During validation, we want to use all instances/patches
                #     # and if its number is very big, we may run out of GPU memory
                #     # in this case, we first iteratively go over subsets of patches to calculate backbone features
                #     # and at the very end calculate the classification output

                #     logits = []
                #     logits2 = []

                #     for i in range(int(np.ceil(data.shape[1] / float(max_tiles)))):
                #         data_slice = data[:, i * max_tiles : (i + 1) * max_tiles]
                #         logits_slice = model(data_slice, no_head=True)
                #         logits.append(logits_slice)

                #         # if has_extra_outputs:
                #         #     logits2.append(
                #         #         [
                #         #             extra_outputs["layer1"],
                #         #             extra_outputs["layer2"],
                #         #             extra_outputs["layer3"],
                #         #             extra_outputs["layer4"],
                #         #         ]
                #         #     )

                #     logits = torch.cat(logits, dim=1)
                #     # if has_extra_outputs:
                #     #     extra_outputs["layer1"] = torch.cat([l[0] for l in logits2], dim=0)
                #     #     extra_outputs["layer2"] = torch.cat([l[1] for l in logits2], dim=0)
                #     #     extra_outputs["layer3"] = torch.cat([l[2] for l in logits2], dim=0)
                #     #     extra_outputs["layer4"] = torch.cat([l[3] for l in logits2], dim=0)

                #     logits = calc_head(logits)

                # else:
                # if number of instances is not big, we can run inference directly
                logits = model(data.squeeze(1))

                loss = criterion(logits, target)

            pred = logits.sigmoid().sum(1).detach().round()
            target = target.sum(1).round()
            acc = (pred == target).float().mean()

            run_loss.append(loss)
            run_acc.append(acc)
            loss = run_loss.aggregate()
            acc = run_acc.aggregate()

            PREDS.extend(pred)
            TARGETS.extend(target)

            # if args.rank == 0:
            #     print(
            #         "Val epoch {}/{} {}/{}".format(epoch, args.epochs, idx, len(loader)),
            #         "loss: {:.4f}".format(loss),
            #         "acc: {:.4f}".format(acc),
            #         "time {:.2f}s".format(time.time() - start_time),
            #     )
            start_time = time.time()

        # Calculate QWK metric (Quadratic Weigted Kappa) https://en.wikipedia.org/wiki/Cohen%27s_kappa
        PREDS = PREDS.get_buffer().cpu().numpy()
        TARGETS = TARGETS.get_buffer().cpu().numpy()
        qwk = cohen_kappa_score(PREDS.astype(np.float64), TARGETS.astype(np.float64), weights="quadratic")

    return loss, acc, qwk


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0):
    """Save checkpoint"""

    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()

    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}

    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


class LabelEncodeIntegerGraded(MapTransform):
    """
    Convert an integer label to encoded array representation of length num_classes,
    with 1 filled in up to label index, and 0 otherwise. For example for num_classes=5,
    embedding of 2 -> (1,1,0,0,0)

    Args:
        num_classes: the number of classes to convert to encoded format.
        keys: keys of the corresponding items to be transformed. Defaults to ``'label'``.
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(
        self,
        num_classes: int,
        keys: KeysCollection = "label",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.num_classes = num_classes

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = int(d[key])

            lz = np.zeros(self.num_classes, dtype=np.float32)
            lz[:label] = 1.0
            # alternative oneliner lz=(np.arange(self.num_classes)<int(label)).astype(np.float32) #same oneliner
            d[key] = lz

        return d


def list_data_collate(batch: collections.abc.Sequence):
    """
    Combine instances from a list of dicts into a single dict, by stacking them along first dim
    [{'image' : 3xHxW}, {'image' : 3xHxW}, {'image' : 3xHxW}...] - > {'image' : Nx3xHxW}
    followed by the default collate which will form a batch BxNx3xHxW
    """

    for i, item in enumerate(batch):
        # print(f"{i} = {item['image'].shape=} >> {item['image'].keys=}")
        data = item[0]
        data["image"] = torch.stack([ix["image"] for ix in item], dim=0)
        # data["patch_location"] = torch.stack([ix["patch_location"] for ix in item], dim=0)
        batch[i] = data
    return default_collate(batch)


def add_channel_dimension(data):
    data["image"] = torch.unsqueeze(data["image"], dim=0)
    return data
import imageio
def load_image(x):
    return {"image": imageio.imread(x["image"]), "label": x['label']}

def main_worker(gpu, args):
    args.gpu = gpu

    if args.distributed:
        args.rank = args.rank * torch.cuda.device_count() + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )

    print(args.rank, " gpu", args.gpu)

    torch.cuda.set_device(args.gpu)  # use this default device (same as args.device if not distributed)
    torch.backends.cudnn.benchmark = True

    if args.rank == 3:
        print("Batch size is:", args.batch_size, "epochs", args.epochs)

    # Remove all paths that don't exist
    import json
    with open(args.dataset_json, 'r') as f:
        data = json.load(f)
    existing_data = {'training': [], 'validation': []}
    for split, paths in data.items():
        for item in paths:
            if os.path.exists(args.data_root + item['image']):
                existing_data[split].append(item)
    with open(args.dataset_json + '2', 'w+') as f:
        json.dump(existing_data, f)            
    args.dataset_json = args.dataset_json + '2'

    #############
    # Create MONAI dataset
    training_list = load_decathlon_datalist(
        data_list_file_path=args.dataset_json,
        data_list_key="training",
        base_dir=args.data_root,
    )
    validation_list = load_decathlon_datalist(
        data_list_file_path=args.dataset_json,
        data_list_key="validation",
        base_dir=args.data_root,
    )

    if args.quick:  # for debugging on a small subset
        training_list = training_list[:16]
        validation_list = validation_list[:16]
    
    import imageio
    train_transform = Compose(
        [
            load_image,
            LabelEncodeIntegerGraded(keys=["label"], num_classes=args.num_classes),
            # RandGridPatchd(
            #     keys=["image"],
            #     patch_size=(args.tile_size, args.tile_size),
            #     num_patches=args.tile_count,
            #     sort_fn="min",
            #     pad_mode=None,
            #     constant_values=255,
            # ),
            SplitDimd(keys=["image"], dim=0, keepdim=False, list_output=True),
            RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
            RandRotate90d(keys=["image"], prob=0.5),
            ScaleIntensityRanged(keys=["image"], a_min=np.float32(0), a_max=np.float32(255)),
            ToTensord(keys=["image", "label"]),
        ]
    )

    valid_transform = Compose(
        [
            load_image,
            LabelEncodeIntegerGraded(keys=["label"], num_classes=args.num_classes),
            # GridPatchd(
            #     keys=["image"],
            #     patch_size=(args.tile_size, args.tile_size),
            #     threshold=0.999 * 3 * 255 * args.tile_size * args.tile_size,
            #     pad_mode=None,
            #     constant_values=255,
            # ),
            SplitDimd(keys=["image"], dim=0, keepdim=False, list_output=True),
            ScaleIntensityRanged(keys=["image"], a_min=np.float32(0), a_max=np.float32(255)),
            ToTensord(keys=["image", "label"]),
        ]
    )

    dataset_train = Dataset(data=training_list, transform=train_transform)
    dataset_valid = Dataset(data=validation_list, transform=valid_transform)

    train_sampler = DistributedSampler(dataset_train) if args.distributed else None
    val_sampler = DistributedSampler(dataset_valid, shuffle=False) if args.distributed else None

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        multiprocessing_context="spawn" if args.workers > 0 else None,
        sampler=train_sampler,
        collate_fn=list_data_collate,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        multiprocessing_context="spawn" if args.workers > 0 else None,
        sampler=val_sampler,
        collate_fn=list_data_collate,
    )

    if args.rank == 3:
        print("Dataset training:", len(dataset_train), "validation:", len(dataset_valid))

    # model = milmodel.MILModel(num_classes=args.num_classes, pretrained=True, mil_mode=args.mil_mode)
    import torch.nn.functional as F
    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 9, kernel_size=1)
            self.bn1 = nn.BatchNorm2d(9, affine=True)
            self.prelu1 = nn.PReLU()

            self.conv2 = nn.Conv2d(9, 27, kernel_size=1)
            self.bn2 = nn.BatchNorm2d(27, affine=True)
            self.prelu2 = nn.PReLU()

            self.conv3 = nn.Conv2d(27, 81, kernel_size=1)
            self.bn3 = nn.BatchNorm2d(81, affine=True)
            self.prelu3 = nn.PReLU()

            self.conv4 = nn.Conv2d(81, 27, kernel_size=3, padding=1, stride=2)
            self.bn4 = nn.BatchNorm2d(27, affine=True)
            self.prelu4 = nn.PReLU()

            self.conv5 = nn.Conv2d(27, 9, kernel_size=3, padding=1, stride=2)
            self.bn5 = nn.BatchNorm2d(9, affine=True)
            self.prelu5 = nn.PReLU()

            self.conv6 = nn.Conv2d(9, 3, kernel_size=3, padding=1, stride=2)
            self.bn6 = nn.BatchNorm2d(3, affine=True)
            self.prelu6 = nn.PReLU()

            self.conv0 = nn.Conv2d(3, 1, kernel_size=3, padding=1, stride=1)
            self.bn0 = nn.BatchNorm2d(1, affine=True)
            self.prelu0 = nn.PReLU()

            self.linear7 = nn.Linear(1 * 128 * 128, 1 * 64 * 64)
            self.bn7 = nn.LayerNorm(64 * 64, elementwise_affine=True)
            self.prelu7 = nn.PReLU()
            self.linear8 = nn.Linear(1 * 64 * 64, 1 * 32 * 32)
            self.bn8 = nn.LayerNorm(32 * 32, elementwise_affine=True)
            self.prelu8 = nn.PReLU()
            self.linear9 = nn.Linear(1 * 32 * 32, 1 * 16 * 16)
            self.bn9 = nn.LayerNorm(16 * 16, elementwise_affine=True)
            self.prelu9 = nn.PReLU()

            self.linear10 = nn.Linear(1 * 16 * 16, 5)

        def forward(self, x):
            # Flatten the input image
            x = self.prelu1(self.bn1(self.conv1(x)))
            x = self.prelu2(self.bn2(self.conv2(x)))
            x = self.prelu3(self.bn3(self.conv3(x)))
            x = self.prelu4(self.bn4(self.conv4(x)))
            x = self.prelu5(self.bn5(self.conv5(x)))
            x = self.prelu6(self.bn6(self.conv6(x)))
            x = self.prelu0(self.bn0(self.conv0(x)))

            x = x.view(x.size(0), -1)
            x = self.prelu7(self.bn7(self.linear7(x)))
            x = self.prelu8(self.bn8(self.linear8(x)))
            x = self.prelu9(self.bn9(self.linear9(x)))
            
            x = self.linear10(x)
            return x
    model = MyModel()

    best_acc = 0
    start_epoch = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

    if args.validate:
        # if we only want to validate existing checkpoint
        epoch_time = time.time()
        val_loss, val_acc, qwk = val_epoch(model, valid_loader, epoch=0, args=args, max_tiles=args.tile_count)
        if args.rank == 3:
            print(
                "Final validation loss: {:.4f}".format(val_loss),
                "acc: {:.4f}".format(val_acc),
                "qwk: {:.4f}".format(qwk),
                "time {:.2f}s".format(time.time() - epoch_time),
            )

        exit(0)

    params = model.parameters()

    # if args.mil_mode in ["att_trans", "att_trans_pyramid"]:
    #     m = model if not args.distributed else model.module
    #     params = [
    #         {"params": list(m.attention.parameters()) + list(m.myfc.parameters()) + list(m.net.parameters())},
    #         {"params": list(m.transformer.parameters()), "lr": 6e-6, "weight_decay": 0.1},
    #     ]

    optimizer = torch.optim.AdamW(params, lr=args.optim_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    if args.logdir is not None and args.rank == 3:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 3:
            print("Writing Tensorboard logs to ", writer.log_dir)
    else:
        writer = None

    # RUN TRAINING
    n_epochs = args.epochs
    val_acc_max = 0.0

    scaler = GradScaler(enabled=args.amp)

    best_acc, best_epoch = 0, 0
    for epoch in tqdm(range(start_epoch, n_epochs)):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            torch.distributed.barrier()

        print(args.rank, time.ctime(), "Epoch:", epoch)

        epoch_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler=scaler, epoch=epoch, args=args)

        if args.rank == 3:
            print(
                "Final training  {}/{}".format(epoch, n_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "acc: {:.4f}".format(train_acc),
                "time {:.2f}s".format(time.time() - epoch_time),
            )

        if args.rank == 3 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("train_acc", train_acc, epoch)

        b_new_best = False
        val_acc = 0
        if (epoch + 1) % args.val_every == 0:
            epoch_time = time.time()
            val_loss, val_acc, qwk = val_epoch(model, valid_loader, epoch=epoch, args=args, max_tiles=args.tile_count)
            if args.rank == 3:
                print(
                    "Final validation  {}/{}".format(epoch, n_epochs - 1),
                    "loss: {:.4f}".format(val_loss),
                    "acc: {:.4f}".format(val_acc),
                    "qwk: {:.4f}".format(qwk),
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_loss", val_loss, epoch)
                    writer.add_scalar("val_acc", val_acc, epoch)
                    writer.add_scalar("val_qwk", qwk, epoch)

                val_acc = qwk

                if val_acc > val_acc_max:
                    print("qwk ({:.6f} --> {:.6f})".format(val_acc_max, val_acc))
                    val_acc_max = val_acc
                    b_new_best = True
                    best_acc = val_acc
                    best_epoch = epoch

        if args.rank == 3 and args.logdir is not None:
            save_checkpoint(model, epoch, args, best_acc=val_acc, filename="model_final.pt")
            if b_new_best:
                print("Copying to model.pt new best model!!!!")
                shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        scheduler.step()

    print(best_acc, best_epoch)
    print("ALL DONE")


def parse_args():
    parser = argparse.ArgumentParser(description="Multiple Instance Learning (MIL) example of classification from WSI.")
    parser.add_argument(
        "--data_root", default="/data/breast-cancer/PANDA/train_images_FFT_parts/", help="path to root folder of images"
    )
    parser.add_argument("--dataset_json", default=None, type=str, help="path to dataset json file")

    parser.add_argument("--num_classes", default=5, type=int, help="number of output classes")
    parser.add_argument("--mil_mode", default="att_trans", help="MIL algorithm")
    parser.add_argument(
        "--tile_count", default=22, type=int, help="number of patches (instances) to extract from WSI image"
    )
    parser.add_argument("--tile_size", default=256, type=int, help="size of square patch (instance) in pixels")

    parser.add_argument("--checkpoint", default=None, help="load existing checkpoint")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="run only inference on the validation set, must specify the checkpoint argument",
    )

    parser.add_argument("--logdir", default=None, help="path to log directory to store Tensorboard logs")

    parser.add_argument("--epochs", "--max_epochs", default=50, type=int, help="number of training epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size, the number of WSI images per gpu")
    parser.add_argument("--optim_lr", default=3e-5, type=float, help="initial learning rate")

    parser.add_argument("--weight_decay", default=0, type=float, help="optimizer weight decay")
    parser.add_argument("--amp", action="store_true", help="use AMP, recommended")
    parser.add_argument(
        "--val_every",
        "--val_interval",
        default=1,
        type=int,
        help="run validation after this number of epochs, default 1 to run every epoch",
    )
    parser.add_argument("--workers", default=1, type=int, help="number of workers for data loading")

    # for multigpu
    parser.add_argument("--distributed", default=False, action="store_true", help="use multigpu training, recommended")
    parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--rank", default=3, type=int, help="node rank for distributed training")
    parser.add_argument(
        "--dist-url", default="tcp://127.0.0.1:23456", type=str, help="url used to set up distributed training"
    )
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")

    parser.add_argument("--quick", default=False, action="store_true", help="use a small subset of data for debugging")

    args = parser.parse_args()

    print("Argument values:")
    for k, v in vars(args).items():
        print(k, "=>", v)
    print("-----------------")

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.dataset_json is None:
        # download default json datalist
        resource = "https://drive.google.com/uc?id=1L6PtKBlHHyUgTE4rVhRuOLTQKgD4tBRK"
        dst = "./datalist_panda_0.json"
        if not os.path.exists(dst):
            gdown.download(resource, dst, quiet=False)
        args.dataset_json = dst

    if args.distributed:
        ngpus_per_node = torch.cuda.device_count()
        args.optim_lr = ngpus_per_node * args.optim_lr / 2  # heuristic to scale up learning rate in multigpu setup
        args.world_size = ngpus_per_node * args.world_size

        print("Multigpu", ngpus_per_node, "rescaled lr", args.optim_lr)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,))
    else:
        main_worker(3, args)