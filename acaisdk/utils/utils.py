import os
import hashlib
from _hashlib import HASH as Hash
from pathlib import Path
from typing import Union

IS_CLI = False

DEBUG = os.environ.get('ACAI_DEBUG', '').lower() in ('1', 'true', 'T')


def bytes_to_size(size):
    """Convert size in bytes into human readable string.

    Args:
        size (int): Size in bytes.

    Return:
        (str): Converted size string.
    """
    if not size >> 10 or size < 0:
        return str(size)
    elif not size >> 20:
        return '{:.2f}KB'.format(size / 1024.0)
    elif not size >> 30:
        return '{:.2f}MB'.format(size / (1024.0 ** 2))
    elif not size >> 40:
        return '{:.2f}GB'.format(size / (1024.0 ** 3))
    else:
        return '{:.2f}TB'.format(size / (1024.0 ** 4))


def md5_update_from_file(filename: Union[str, Path], hash: Hash):
    assert Path(filename).is_file()
    hash.update(filename.encode())
    with open(str(filename), "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash.update(chunk)
    return hash

    
def md5_update_from_file_list(filelist: [Union[str, Path]], hash: Hash):
    for file in sorted(filelist):
        hash = md5_update_from_file(file, hash)
    return hash


def md5_file(filename: Union[str, Path]):
    return str(md5_update_from_file(filename, hashlib.md5()).hexdigest())


def md5_file_list(filelist: [Union[str, Path]]):
    return str(md5_update_from_file_list(filelist, hashlib.md5()).hexdigest())
    

def debug(*msg, newline=True):
    if DEBUG:
        print(*msg, end='\n' if newline else '')
