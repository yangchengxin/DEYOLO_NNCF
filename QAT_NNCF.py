import sys
import time
import warnings  # To disable warnings on export model
import zipfile
from pathlib import Path

import torch

import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import openvino as ov
from torch.jit import TracerWarning

sys.path.append("../utils")
from notebook_utils import download_file

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

MODEL_DIR = Path("model")
OUTPUT_DIR = Path("output")
DATA_DIR = Path("data")
BASE_MODEL_NAME = "resnet18"
image_size = 64

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Paths where PyTorch and OpenVINO IR models will be stored.
fp32_pth_path = Path(MODEL_DIR / (BASE_MODEL_NAME + "_fp32")).with_suffix(".pth")
fp32_ir_path = fp32_pth_path.with_suffix(".xml")
int8_ir_path = Path(MODEL_DIR / (BASE_MODEL_NAME + "_int8")).with_suffix(".xml")

# It is possible to train FP32 model from scratch, but it might be slow. Therefore, the pre-trained weights are downloaded by default.
pretrained_on_tiny_imagenet = True
fp32_pth_url = "https://storage.openvinotoolkit.org/repositories/nncf/openvino_notebook_ckpts/302_resnet18_fp32_v1.pth"
download_file(fp32_pth_url, directory=MODEL_DIR, filename=fp32_pth_path.name)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def download_tiny_imagenet_200(
        data_dir: Path,
        url="http://cs231n.stanford.edu/tiny-imagenet-200.zip",
        tarname="tiny-imagenet-200.zip",
):
    archive_path = data_dir / tarname
    download_file(url, directory=data_dir, filename=tarname)
    zip_ref = zipfile.ZipFile(archive_path, "r")
    zip_ref.extractall(path=data_dir)
    zip_ref.close()


def prepare_tiny_imagenet_200(dataset_dir: Path):
    # Format validation set the same way as train set is formatted.
    val_data_dir = dataset_dir / 'val'
    val_annotations_file = val_data_dir / 'val_annotations.txt'
    with open(val_annotations_file, 'r') as f:
        val_annotation_data = map(lambda line: line.split('\t')[:2], f.readlines())
    val_images_dir = val_data_dir / 'images'
    for image_filename, image_label in val_annotation_data:
        from_image_filepath = val_images_dir / image_filename
        to_image_dir = val_data_dir / image_label
        if not to_image_dir.exists():
            to_image_dir.mkdir()
        to_image_filepath = to_image_dir / image_filename
        from_image_filepath.rename(to_image_filepath)
    val_annotations_file.unlink()
    val_images_dir.rmdir()


DATASET_DIR = DATA_DIR / "tiny-imagenet-200"
if not DATASET_DIR.exists():
    download_tiny_imagenet_200(DATA_DIR)
    prepare_tiny_imagenet_200(DATASET_DIR)
    print(f"Successfully downloaded and prepared dataset at: {DATASET_DIR}")


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter("Time", ":3.3f")
    losses = AverageMeter("Loss", ":2.3f")
    top1 = AverageMeter("Acc@1", ":2.2f")
    top5 = AverageMeter("Acc@5", ":2.2f")
    progress = ProgressMeter(
        len(train_loader), [batch_time, losses, top1, top5], prefix="Epoch:[{}]".format(epoch)
    )

    # Switch to train mode.
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)

        # Compute output.
        output = model(images)
        loss = criterion(output, target)

        # Measure accuracy and record loss.
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # Compute gradient and do opt step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time.
        batch_time.update(time.time() - end)
        end = time.time()

        print_frequency = 50
        if i % print_frequency == 0:
            progress.display(i)

def validate(val_loader, model, criterion):
    batch_time = AverageMeter("Time", ":3.3f")
    losses = AverageMeter("Loss", ":2.3f")
    top1 = AverageMeter("Acc@1", ":2.2f")
    top5 = AverageMeter("Acc@5", ":2.2f")
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix="Test: ")

    # Switch to evaluate mode.
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # Compute output.
            output = model(images)
            loss = criterion(output, target)

            # Measure accuracy and record loss.
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # Measure elapsed time.
            batch_time.update(time.time() - end)
            end = time.time()

            print_frequency = 10
            if i % print_frequency == 0:
                progress.display(i)

        print(" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5))
    return top1.avg