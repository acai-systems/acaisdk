import os
import signal
import multiprocessing

from tqdm import tqdm
from acaisdk.utils.rest_utils import get_session
from acaisdk.utils import utils


class FileIO:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_size = self.get_file_size(file_path)  # type: int
        self.read_bytes = 0  # type: int

    @staticmethod
    def get_file_size(path):
        return os.path.getsize(path)

    def __upload_old(self, presigned_link: str):
        """Deprecated, use upload() instead. This is not portable."""
        file_object = open(self.file_path, 'rb')
        pid = os.fork()
        if pid == 0:
            try:
                prev_pos = 0
                with tqdm(total=self.file_size,
                          unit='B', unit_scale=True, unit_divisor=1024,
                          ascii=True, disable=not utils.IS_CLI) as p_bar:
                    while 1:
                        offset = file_object.tell()
                        p_bar.update((offset - prev_pos))
                        prev_pos = offset
                        if offset >= self.file_size:
                            break
            finally:
                os._exit(0)

        headers = {'Content-Type': 'application/binary'}
        try:
            r = get_session().put(presigned_link, data=file_object,
                                  headers=headers)
            if file_object.tell() == 0:
                # Hack, somehow S3 presigned url upload does not recognize
                # fd for an empty file but can upload with data=None.
                r = get_session().put(presigned_link, data=None,
                                      headers=headers)
            return r
        finally:
            file_object.close()
            os.kill(pid, signal.SIGKILL)

    def upload(self, presigned_link: str):
        def progress_bar(file_obj, file_size):
            prev_pos = 0
            with tqdm(total=file_size,
                      unit='B', unit_scale=True, unit_divisor=1024,
                      ascii=True, disable=not utils.IS_CLI) as p_bar:
                while 1:
                    offset = file_obj.tell()
                    p_bar.update((offset - prev_pos))
                    prev_pos = offset
                    if offset >= file_size:
                        break

        file_object = open(self.file_path, 'rb')

        p = multiprocessing.Process(target=progress_bar, args=(file_object,
                                                               self.file_size))
        p.start()

        headers = {'Content-Type': 'application/binary'}
        try:
            r = get_session().put(presigned_link, data=file_object,
                                  headers=headers)
            if file_object.tell() == 0:
                # Hack, somehow S3 presigned url upload does not recognize
                # fd for an empty file but can upload with data=None.
                r = get_session().put(presigned_link, data=None,
                                      headers=headers)
            return r
        finally:
            file_object.close()
            p.terminate()
            p.join()

    @staticmethod
    def download(presigned_link: str,
                 local_file_path: str):
        r = get_session().get(presigned_link, verify=False, stream=True)
        r.raise_for_status()
        content_len = int(r.headers['Content-Length'])

        with open(local_file_path, 'wb') as f, tqdm(
                total=content_len,
                unit='B', unit_scale=True,
                unit_divisor=1024,
                ascii=True, disable=not utils.IS_CLI) as p_bar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    p_bar.update(len(chunk))
                    f.write(chunk)
        return r
