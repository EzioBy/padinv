# python3.7
"""Main function to evaluate the similarity between two datasets.

Available metrics:

- test_fid: Calculate the Frechet Inception Distance (FID) between two datasets,
    lower is better.
- test_kid: Calculate the Kernel Inception Distance (KID) between two datasets,
    lower is better.

NOTE: Unlike `test_metrics.py`, this file only supports evaluating two image
datasets without involving pre-trained generators.
"""

import argparse
import os

import torch

from datasets import build_dataset
from metrics import build_metric
from metrics.utils import compute_fid_from_feature
from metrics.utils import compute_kid_from_feature
from utils.loggers import build_logger
from utils.parsing_utils import parse_bool
from utils.dist_utils import init_dist
from utils.dist_utils import exit_dist


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Run metric test.')
    parser.add_argument('--src_dataset', type=str, required=True,
                        help='Path to the source dataset (e.g., the images to '
                             'be improved from) used for metric computation, '
                             'can be a zip file, a tar file, or a directory.')
    parser.add_argument('--tgt_dataset', type=str, required=True,
                        help='Path to the target dataset (e.g., the images '
                             'treated as ground truth or goal) used for metric '
                             'computation, can be a zip file, a tar file, or a '
                             'directory.')
    parser.add_argument('--resolution', type=int, required=True,
                        help='Resolution to evaluation for both source dataset '
                             'and target dataset.')
    parser.add_argument('--src_num', type=int, default=-1,
                        help='Number of source data used for testing. Negative '
                             'means using all data. (default: %(default)s)')
    parser.add_argument('--tgt_num', type=int, default=-1,
                        help='Number of target data used for testing. Negative '
                             'means using all data. (default: %(default)s)')
    parser.add_argument('--image_channels', type=int, default=3,
                        help='Number of channels of the input image. '
                             '(default: %(default)s)')
    parser.add_argument('--work_dir', type=str,
                        default='work_dirs/metric_tests',
                        help='Working directory for metric test. (default: '
                             '%(default)s)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size used for metric computation. '
                             '(default: %(default)s)')
    parser.add_argument('--test_all', type=parse_bool, default=False,
                        help='Whether to run all evaluations. '
                             '(default: %(default)s)')
    parser.add_argument('--test_fid', type=parse_bool, default=False,
                        help='Whether to test FID. (default: %(default)s)')
    parser.add_argument('--test_kid', type=parse_bool, default=False,
                        help='Whether to test KID. (default: %(default)s)')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Replica rank on the current node. This field is '
                             'required by `torch.distributed.launch`. '
                             '(default: %(default)s)')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Initialize distributed environment.
    backend = 'gloo' if os.name == 'nt' else 'nccl'
    init_dist(launcher='pytorch', backend=backend)

    # CUDNN settings.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Dataset settings.
    data_transform_kwargs = dict(
        image_size=args.resolution, image_channels=args.image_channels)
    dataset_kwargs = dict(dataset_type='ImageDataset',
                          annotation_path=None,
                          annotation_meta=None,
                          mirror=False,
                          transform_kwargs=data_transform_kwargs)
    data_loader_kwargs = dict(data_loader_type='iter',
                              repeat=1,
                              num_workers=4,
                              prefetch_factor=2,
                              pin_memory=True)

    src_dataset_kwargs = dataset_kwargs.copy()
    src_dataset_kwargs.update(root_dir=args.src_dataset,
                              max_samples=args.src_num)
    src_data_loader = build_dataset(
        for_training=False,
        batch_size=args.batch_size,
        dataset_kwargs=src_dataset_kwargs,
        data_loader_kwargs=data_loader_kwargs.copy()
    )

    tgt_dataset_kwargs = dataset_kwargs.copy()
    tgt_dataset_kwargs.update(root_dir=args.tgt_dataset,
                              max_samples=args.tgt_num)
    tgt_data_loader = build_dataset(
        for_training=False,
        batch_size=args.batch_size,
        dataset_kwargs=tgt_dataset_kwargs,
        data_loader_kwargs=data_loader_kwargs.copy()
    )

    if torch.distributed.get_rank() == 0:
        logger = build_logger('normal', logfile=None, verbose_log=True)
    else:
        logger = build_logger('dummy')

    src_num = (len(src_data_loader.dataset)
               if args.src_num <= 0 else args.src_num)
    tgt_num = (len(tgt_data_loader.dataset)
               if args.tgt_num <= 0 else args.tgt_num)

    if args.test_all or args.test_fid:
        logger.info('========== Test FID ==========')
        metric = build_metric('FID',
                              name=f'fid_src{src_num}_tgt{tgt_num}',
                              work_dir=args.work_dir,
                              logger=logger,
                              batch_size=args.batch_size,
                              real_num=tgt_num,
                              fake_num=src_num)
        src_feature = metric.extract_real_features(src_data_loader)
        tgt_feature = metric.extract_real_features(tgt_data_loader)
        logger.info(f'Computing {metric.name}, this may take some time...')
        if metric.is_chief:
            val = compute_fid_from_feature(src_feature, tgt_feature)
            result = {metric.name: val}
        else:
            assert src_feature is None and tgt_feature is None
            result = None
        metric.sync()
        metric.save(result)
    if args.test_all or args.test_kid:
        logger.info('========== Test KID ==========')
        metric = build_metric('KID',
                              name=f'kid_src{src_num}_tgt{tgt_num}',
                              work_dir=args.work_dir,
                              logger=logger,
                              batch_size=args.batch_size,
                              real_num=tgt_num,
                              fake_num=src_num)
        src_feature = metric.extract_real_features(src_data_loader)
        tgt_feature = metric.extract_real_features(tgt_data_loader)
        logger.info(f'Computing {metric.name}, this may take some time...')
        if metric.is_chief:
            val = compute_kid_from_feature(src_feature, tgt_feature)
            result = {metric.name: val}
        else:
            assert src_feature is None and tgt_feature is None
            result = None
        metric.sync()
        metric.save(result)

    # Exit distributed environment.
    exit_dist()


if __name__ == '__main__':
    main()
