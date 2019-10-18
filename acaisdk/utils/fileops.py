import os
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

    def upload(self, presigned_link: str):
        file_object = open(self.file_path, 'rb')
        pid = os.fork()
        if pid == 0:
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
            os._exit(0)

        headers = {'Content-Type': 'application/binary'}
        try:
            r = get_session().put(presigned_link, data=file_object,
                                  headers=headers)
            return r
        finally:
            file_object.close()

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
