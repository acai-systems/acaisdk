import os

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


def debug(*msg, newline=True):
    if DEBUG:
        print(*msg, end='\n' if newline else '')
