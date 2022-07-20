# python3.7
"""Collects all runners."""

from .stylegan_runner import StyleGANRunner
from .stylegan2_runner import StyleGAN2Runner
from .stylegan3_runner import StyleGAN3Runner
from .padinv_runner import PadInvRunner

__all__ = ['build_runner']

_RUNNERS = {
    'StyleGANRunner': StyleGANRunner,
    'StyleGAN2Runner': StyleGAN2Runner,
    'StyleGAN3Runner': StyleGAN3Runner,
    'PadInvRunner': PadInvRunner
}


def build_runner(config):
    """Builds a runner with given configuration.

    Args:
        config: Configurations used to build the runner.

    Raises:
        ValueError: If the `config.runner_type` is not supported.
    """
    if not isinstance(config, dict) or 'runner_type' not in config:
        raise ValueError('`runner_type` is missing from configuration!')

    runner_type = config['runner_type']
    if runner_type not in _RUNNERS:
        raise ValueError(f'Invalid runner type: `{runner_type}`!\n'
                         f'Types allowed: {list(_RUNNERS)}.')
    return _RUNNERS[runner_type](config)
