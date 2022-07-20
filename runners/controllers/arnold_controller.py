# python3.7
"""Contains the running controller to interact with Arnold training platform."""

import os
import shutil
import tarfile
from collections import deque

from utils.tf_utils import import_tb_writer
from utils.misc import MEDIA_EXTENSIONS
from utils.misc import check_file_ext
from .base_controller import BaseController

SummaryWriter = import_tb_writer()

__all__ = ['ArnoldController']


class ArnoldController(BaseController):
    """Defines the running to interact with Arnold training platform.

    Arnold clean-up settings:

    - keep_archive_num: How many recent Arnold archives to keep. If set to -1,
        all archives will be kept on HDFS. (default: 20)

    NOTE:
        This controller is set to 'FINAL' priority by default and will only be
        executed on the chief worker.
    """

    def __init__(self, config):
        config.setdefault('priority', 'FINAL')
        config.setdefault('chief_only', True)
        super().__init__(config)

        # Useful info.
        self._output_dir = os.environ.get('ARNOLD_OUTPUT', '')
        self._cmd_dir = os.environ.get('ARNOLD_CMD_DIR', '')
        self._bootstrap = os.environ.get('BOOTSTRAP', '')
        self._working_dir = os.environ.get('CONTAINER_CWD', '')
        self._cudnn_version = os.environ.get('CUDNN_VERSION', '')

        # Misc info.
        self._task_id = os.environ.get('ARNOLD_TASK_ID', '')
        self._task_name = os.environ.get('ARNOLD_TASK_NAME', '')

        # Configs.
        self._hdfs_upload_path = config.get('hdfs_upload_path', '')
        if isinstance(self._hdfs_upload_path, str):
            self._hdfs_upload_path = self._hdfs_upload_path.strip()
        self._use_local_tb = config.get('use_local_tb', False)

        # Arnold archive clean-up options.
        if self.use_arnold() and self._hdfs_upload_path:
            self._keep_archive_num = config.get('keep_archive_num', 20)
            if self._keep_archive_num is None or self._keep_archive_num == 0:
                self._keep_archive_num = -1
            if self._keep_archive_num < 0:
                self.archive_queue = deque(maxlen=0)  # Save memory.
            else:
                self.archive_queue = deque(maxlen=self._keep_archive_num)

    @property
    def output_dir(self):
        """The output directory declared automatically in Arnold environment.

        NOTE: To use Arnold TensorBoard, please redirect the TensorBoard writer
        to this field.
        """
        return self._output_dir

    @property
    def cmd_dir(self):
        """The root directory (absolute) of the code on Arnold."""
        return self._cmd_dir

    @property
    def working_dir(self):
        """The working directory (absolute) on Arnold.

        NOTE: This field is usually different from the path to the code. If you
        hardcode relative path '.' in the program, it always matches this field.
        """
        return self._working_dir

    @property
    def bootstrap(self):
        """The path (absolute) to the entrypoint script in Arnold Task"""
        return self._bootstrap

    @property
    def code_path(self):
        """The same as `self.bootstrap`."""
        return self._bootstrap

    @staticmethod
    def use_arnold():
        """Whether the job is running on Arnold server."""
        return os.environ.get('ARNOLD_OUTPUT', None) is not None

    def setup(self, runner):
        if not self.use_arnold():
            runner.logger.info(
                'Arnold controller takes no effect', indent_level=2)
            return

        runner.logger.info('Arnold settings:', indent_level=2)
        if self._hdfs_upload_path:
            # It is HDFS's responsibility to grant the permission to `job_name`.
            self._hdfs_upload_path = os.path.join(self._hdfs_upload_path,
                                                  runner.job_name)
            runner.logger.info(
                f'Set HDFS upload path as {self._hdfs_upload_path}',
                indent_level=3)
            runner.ft.make_remote_dir(self._hdfs_upload_path)
        else:
            runner.logger.info(
                'Do not upload archive files to HDFS', indent_level=3)

        # Redirect TensorBoard event directory to Arnold.
        # NOTE: To support writing to HDFS-based TensorBoard, TensorFlow is
        # also required in the environment.
        if not self._use_local_tb and runner.tb_writer is not None:
            runner.logger.info(
                f'TensorBoard events are redirected to {self.output_dir}',
                indent_level=3)
            runner.logger.warning('Writing to remote TensorBoard is known to '
                                  'increase the memory footprint during '
                                  'training.', indent_level=3)
            runner.tb_writer.close()
            runner.tb_writer = SummaryWriter(log_dir=self.output_dir)
            shutil.rmtree(runner.tensorboard_dir)

        if self.use_arnold() and self._hdfs_upload_path:
            if self.archive_queue.maxlen > 0:
                runner.logger.info(
                    f'Keep at most {self._keep_archive_num} archives',
                    indent_level=3)
            else:
                runner.logger.info('Keep all archives', indent_level=3)
        super().setup(runner)

    def require_clean_up(self):
        """Returns whether the outdated archived file should be removed."""
        if self.archive_queue.maxlen == 0:
            return False
        return len(self.archive_queue) == self.archive_queue.maxlen

    def clean_up(self, runner):
        """Removes the outdated archived file."""
        hdfs_path = self.archive_queue.popleft()  # Pop out the outdated file.
        runner.logger.info(f'Remove {hdfs_path} in the routine clean-up '
                           f'of {self.name}.')
        runner.ft.remove(hdfs_path)

    def archive(self, runner):
        """Archives the intermediate results to HDFS."""
        assert runner.is_chief

        def skip_archive(directory, filenames):
            """Determines whether a file should skip archive."""
            if directory == runner.work_dir:
                # Skip copying data.
                return [os.path.basename(runner.data_dir)]
            if directory == runner.result_dir:
                # Skip copying `numpy` result files.
                return [n for n in filenames if check_file_ext(n, '.npy')]
            return []

        # Archive files in working directory to a temporary directory.
        archive_dir = f'{runner.work_dir}_archive'
        shutil.copytree(runner.work_dir, archive_dir, ignore=skip_archive)
        # Compress archived files.
        archive_name = f'{runner.job_name}-{runner.iter:06d}'
        archive_filename = f'{archive_name}.tar.gz'
        archive_filepath = os.path.join(runner.work_dir, archive_filename)
        with tarfile.open(archive_filepath, 'w:gz') as tar:
            tar.add(archive_dir, arcname=archive_name, recursive=True)
        runner.ft.push(archive_filepath, self._hdfs_upload_path)
        hdfs_path = os.path.join(self._hdfs_upload_path, archive_filename)
        self.archive_queue.append(hdfs_path)
        # Delete archived files and temporary directory.
        os.remove(archive_filepath)
        shutil.rmtree(archive_dir)

        # Archive log and evaluation metrics separately.
        logfile_name = os.path.basename(runner.log_path)
        hdfs_path = os.path.join(self._hdfs_upload_path,
                                 f'{runner.job_name}-{logfile_name}')
        runner.ft.remove(hdfs_path)
        runner.ft.push(runner.log_path, hdfs_path)
        for filename in os.listdir(runner.result_dir):
            if check_file_ext(filename, '.txt'):
                filepath = os.path.join(runner.result_dir, filename)
                hdfs_path = os.path.join(self._hdfs_upload_path,
                                         f'{runner.job_name}-{filename}')
                runner.ft.remove(hdfs_path)
                runner.ft.push(filepath, hdfs_path)

        # Delete checkpoints and results that have already been archived.
        for filename in os.listdir(runner.checkpoint_dir):
            if not filename.startswith('best-'):  # Skip best checkpoints.
                filepath = os.path.join(runner.checkpoint_dir, filename)
                os.remove(filepath)
        for filename in os.listdir(runner.result_dir):
            if check_file_ext(filename, *MEDIA_EXTENSIONS):
                filepath = os.path.join(runner.result_dir, filename)
                os.remove(filepath)

    def execute_after_iteration(self, runner):
        if not self.use_arnold() or not self._hdfs_upload_path:
            return

        if self.require_clean_up():
            self.clean_up(runner)
        self.archive(runner)
