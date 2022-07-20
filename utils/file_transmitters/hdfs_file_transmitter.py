# python3.7
"""Contains the class of HDFS file transmitter.

The transmitter builds the connection between the local file system to HDFS.

Command settings:

- cmd_prefix: Command prefix, currently can be `doas hdfs dfs` or `hdfs dfs`.
"""

from utils.misc import print_and_execute
from .base_file_transmitter import BaseFileTransmitter

__all__ = ['HDFSFileTransmitter']


class HDFSFileTransmitter(BaseFileTransmitter):
    """Implements the transmitter connecting local file system to HDFS."""

    def __init__(self, cmd_prefix='hdfs dfs'):
        super().__init__()
        self.cmd_prefix = cmd_prefix

    def download_hard(self, src, dst):
        print_and_execute(f'{self.cmd_prefix} -get {src} {dst}')

    def download_soft(self, src, dst):
        print_and_execute(f'{self.cmd_prefix} -get {src} {dst}')

    def upload(self, src, dst):
        print_and_execute(f'{self.cmd_prefix} -put {src} {dst}')

    def delete(self, path):
        print_and_execute(f'{self.cmd_prefix} -rm -r {path}')

    def make_remote_dir(self, directory):
        print_and_execute(f'{self.cmd_prefix} -mkdir -p {directory}')
