# python3.7
"""Contains the class to evaluate GAN inversion.

The metrics include MSE (mean square error), LPIPS (Learned Perceptual Image
Patch Similarity) and IDSIM (identity similarity). This class also supports
visualizing the input image and the inversion result for qualitative evaluation.

LPIPS metric is introduced in paper:

https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.pdf

Identity similarity (IDSIM) is computed based on the feature extracted from
ArcFace face recognition model, and cosine similarity is used.

ArcFace is introduced in paper:

https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html
"""

import os.path
import time

import torch
import torch.nn.functional as F

from models import build_model
from models.lpips.lpips import LPIPS
from utils.visualizers import GridVisualizer
from utils.image_utils import postprocess_image
from .base_metric import BaseMetric

__all__ = ['GANInversionMetric']


class GANInversionMetric(BaseMetric):
    """Defines the class for GAN inversion metrics computation."""

    def __init__(self,
                 name='gan_inversion',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 calc_mse=True,
                 calc_lpips=True,
                 lpips_net='alex',
                 calc_idsim=True,
                 viz_num=8,
                 min_val=-1.0,
                 max_val=1.0):
        """Initializes the class with desired metrics.

        NOTE: `batch_size` takes no effect, which will always be determined by
        the data loader.

        Args:
            lpips_net: The network architecture of LPIPS. (default: alex).
                       Note that we provide 2 versions of LPIPS implementation.
                       Feel free to choose one.
            calc_mse: Whether to compute MSE. (default: True)
            calc_lpips: Whether to compute LPIPS. (default: True)
            calc_idsim: Whether to compute IDSIM. (default: True)
            viz_num: Number of inversion results for visualization. Use `0` to
                disable. (default: 8)
            min_val: Minimum pixel value of the synthesized images. This field
                is particularly used for image visualization. (default: -1.0)
            max_val: Maximum pixel value of the synthesized images. This field
                is particularly used for image visualization. (default: 1.0)
        """
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size)
        self.calc_mse = calc_mse
        self.calc_lpips = calc_lpips
        self.lpips_net = lpips_net
        self.calc_idsim = calc_idsim
        self.viz_num = viz_num
        self.min_val = min_val
        self.max_val = max_val
        self.requires_test = calc_mse or calc_lpips or calc_idsim or viz_num > 0
        assert self.lpips_net in ['alex', 'vgg'], 'Currently we support alex and vgg as LPIPS backbones.'

        # Build perceptual model for feature extraction.
        if self.calc_lpips:
            if self.lpips_net == 'alex':
                self.lpips_loss = LPIPS(net_type='alex').cuda().eval()
            elif self.lpips_net == 'vgg':
                self.perceptual_model = build_model('PerceptualModel')

        # Build ArcFace model for identity-related feature extraction.
        if self.calc_idsim:
            self.arcface_model = build_model('ArcFaceModel')

        # Initialize visualizer.
        if self.viz_num > 0:
            self.visualizer = GridVisualizer(col_spacing=10)

    def evaluate(self,
                 data_loader,
                 generator,
                 generator_kwargs,
                 encoder,
                 encoder_kwargs):
        if not self.requires_test:
            self.sync()
            return None

        total_num = len(data_loader.dataset)

        G = generator
        G_kwargs = generator_kwargs
        G_mode = G.training  # save model training mode.
        G.eval()
        E = encoder
        E_kwargs = encoder_kwargs
        E_mode = E.training  # save model training mode.
        E.eval()

        self.logger.info(f'Inference encoder {self.log_tail}.', is_verbose=True)
        self.logger.init_pbar()
        pbar_task = self.logger.add_pbar_task('Num samples', total=total_num)
        all_mse = []
        all_lpips = []
        all_idsim = []
        all_viz = []
        batch_size = data_loader.batch_size
        replica_num = self.get_replica_num(total_num)
        for batch_idx in range(len(data_loader)):
            if batch_idx * batch_size >= replica_num:
                # NOTE: Here, we always go through the entire dataset to make
                # sure the next evaluator can visit the data loader from the
                # beginning.
                _batch_data = next(data_loader)
                continue
            with torch.no_grad():
                # Forward encoder and generator for image reconstruction.
                batch_src = next(data_loader)['image'].cuda().detach()
                batch_latents = E(batch_src, **E_kwargs)
                batch_rec = G.synthesis(
                    batch_latents['style_latents'],
                    batch_latents['padding_maps'],
                    **G_kwargs)['image']
                if batch_src.size() != batch_rec.size():
                    batch_rec = F.interpolate(batch_rec,
                                              size=batch_src.shape[-2:])
                # Compute MSE if needed.
                if self.calc_mse:
                    batch_mse = (batch_src - batch_rec).square().mean((1, 2, 3))
                    gathered_mse = self.gather_batch_results(batch_mse)
                    self.append_batch_results(gathered_mse, all_mse)
                # Compute LPIPS if needed.
                if self.calc_lpips:
                    if self.lpips_net == 'alex':
                        batch_lpips = self.lpips_loss(batch_src, batch_rec)
                        batch_lpips = batch_lpips.unsqueeze(0)
                    elif self.lpips_net == 'vgg':
                        batch_lpips = self.perceptual_model(batch_src, batch_rec, return_tensor='lpips')
                    gathered_lpips = self.gather_batch_results(batch_lpips)
                    self.append_batch_results(gathered_lpips, all_lpips)
                # Compute IDSIM if needed.
                if self.calc_idsim:
                    src_features = self.arcface_model(batch_src)
                    rec_features = self.arcface_model(batch_rec)
                    # Features have already been normalized by ArcFace model.
                    batch_idsim = (src_features * rec_features).sum(dim=1)
                    gathered_idsim = self.gather_batch_results(batch_idsim)
                    self.append_batch_results(gathered_idsim, all_idsim)
                # Save visualization results if needed.
                if (batch_idx + 1) * batch_size <= self.viz_num:
                    # Concatenate images side-by-side (i.e., width dimension).
                    batch_viz = torch.cat((batch_src, batch_rec), dim=3)
                    gathered_viz = self.gather_batch_results(batch_viz)
                    self.append_batch_results(gathered_viz, all_viz)
            self.logger.update_pbar(pbar_task, batch_size * self.world_size)
        self.logger.close_pbar()

        if self.is_chief:
            result = dict()
            if self.calc_mse:
                all_mse = self.gather_all_results(all_mse)[:total_num]
                assert all_mse.shape == (total_num,)
                result[f'{self.name}_mse'] = float(all_mse.mean())
            if self.calc_lpips:
                all_lpips = self.gather_all_results(all_lpips)[:total_num]
                if self.lpips_net != 'alex':
                    assert all_lpips.shape == (total_num,)
                result[f'{self.name}_lpips'] = float(all_lpips.mean())
            if self.calc_idsim:
                all_idsim = self.gather_all_results(all_idsim)[:total_num]
                assert all_idsim.shape == (total_num,)
                result[f'{self.name}_idsim'] = float(all_idsim.mean())
            if self.viz_num > 0:
                viz_num = min(self.viz_num, total_num)
                all_viz = self.gather_all_results(all_viz)[:viz_num]
                assert all_viz.ndim == 4 and all_viz.shape[0] == viz_num
                result[f'{self.name}_viz'] = all_viz
        else:
            assert len(all_mse) == 0
            assert len(all_lpips) == 0
            assert len(all_idsim) == 0
            assert len(all_viz) == 0
            result = None

        if G_mode:
            G.train()  # restore model training mode.
        if E_mode:
            E.train()  # restore model training mode.

        self.sync()
        return result

    def _is_better_than(self, metric_name, new, ref):
        """Performance comparison.

        Lower MSE means better reconstruction.
        Lower LPIPS means better perceptual similarity.
        Higher IDSIM means better identity similarity.
        """
        if metric_name == f'{self.name}_mse':
            return ref is None or new < ref
        if metric_name == f'{self.name}_lpips':
            return ref is None or new < ref
        if metric_name == f'{self.name}_idsim':
            return ref is None or new > ref
        return None

    def save(self, result, target_filename=None, log_suffix=None, tag=None):
        if not self.is_chief or not self.requires_test:
            assert result is None
            self.sync()
            return

        assert isinstance(result, dict)
        msg = f'Evaluating `{self.name}`: '
        if self.calc_mse:
            mse = result[f'{self.name}_mse']
            assert isinstance(mse, float)
            msg += f'MSE {mse:.3e}, '
        if self.calc_lpips:
            lpips = result[f'{self.name}_lpips']
            assert isinstance(lpips, float)
            msg += f'LPIPS: {lpips:.3e}, '
        if self.calc_idsim:
            idsim = result[f'{self.name}_idsim']
            assert isinstance(idsim, float)
            msg += f'IDSIM: {idsim:.3e}, '
        if self.viz_num > 0:
            images = result[f'{self.name}_viz']
            images = postprocess_image(
                images, min_val=self.min_val, max_val=self.max_val)
            filename = target_filename or f'{self.name}_viz_results'
            save_path = os.path.join(self.work_dir, f'{filename}.png')
            self.visualizer.visualize_collection(images, save_path)
            msg += f'visualizing {images.shape[0]} samples, '
        if log_suffix is None:
            msg = msg[:-2] + '.'
        else:
            msg = msg + log_suffix + '.'
        self.logger.info(msg)

        with open(os.path.join(self.work_dir, f'{self.name}.txt'), 'a+') as f:
            date = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'[{date}] {msg}\n')

        # Save to TensorBoard if needed.
        if self.tb_writer is not None:
            if tag is None:
                self.logger.warning('`Tag` is missing when writing data to '
                                    'TensorBoard, hence, the data may be '
                                    'mixed up!')
            if self.calc_mse:
                self.tb_writer.add_scalar(f'Metrics/{self.name}_mse', mse, tag)
            if self.calc_lpips:
                self.tb_writer.add_scalar(
                    f'Metrics/{self.name}_lpips', lpips, tag)
            if self.calc_idsim:
                self.tb_writer.add_scalar(
                    f'Metrics/{self.name}_idsim', idsim, tag)
            if self.viz_num:
                self.tb_writer.add_image(
                    self.name, self.visualizer.grid, tag, dataformats='HWC')
            self.tb_writer.flush()
        self.sync()
