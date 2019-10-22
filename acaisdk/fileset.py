from acaisdk.services.api_calls import *
from typing import Iterable, List
from acaisdk import file


class FilesList(list):
    def as_new_file_set(self, file_set_name):
        return FileSet.create_file_set(file_set_name,
                                       [_[1] for _ in self])

    def upload(self):
        return file.File.upload(self)


class FileSet:
    @staticmethod
    def create_file_set(file_set_name: str,
                        remote_file_list: Iterable):
        """
        :return:    {
                      "id": "HotpotQA:1",
                      "files": [
                        "data/train.json:2"
                      ]
                    }
        """
        data = {
            "name": file_set_name,
            "files": list(remote_file_list)
        }
        return RestRequest(StorageApi.create_file_set) \
            .with_data(data) \
            .with_credentials() \
            .run()

    @staticmethod
    def list_file_set_content(vague_name) -> dict:
        params = {'vague_name': vague_name}
        return RestRequest(StorageApi.resolve_file_set) \
            .with_query(params) \
            .with_credentials() \
            .run()

    @staticmethod
    def resolve_vague_name(vague_name):
        return file.File.resolve_vague_path(vague_name)

    @staticmethod
    def download_file_set(vague_name: str,
                          mount_point: str = None,
                          force: bool = False) -> None:
        if not mount_point:
            # Use current directory
            mount_point = os.getcwd()
        else:
            mount_point = os.path.abspath(mount_point)

        # Get file set content
        params = {'vague_name': vague_name}
        r = RestRequest(StorageApi.download_file_set) \
            .with_query(params) \
            .with_credentials() \
            .run()

        local_paths = []
        remote_paths = []
        urls = []
        for d in r['files']:
            remote_path = d['path'].strip()
            remote_paths.append(remote_path)
            urls.append(d['url'])

            if remote_path.startswith('/'):
                remote_path = '.' + remote_path
            local_path = os.path.join(mount_point, remote_path)
            local_paths.append(os.path.normpath(local_path))

        for path in local_paths:
            # Validate if there's collision
            if not force and os.path.exists(path):
                raise AcaiException('{} already exists. '
                                    'Use "force" to enable overwriting'
                                    ''.format(path))
            folder = os.path.dirname(path)
            os.makedirs(folder, exist_ok=True)

        download_request = {r_path: l_path for r_path, l_path
                            in zip(remote_paths, local_paths)}
        file.File.download(download_request)

    @staticmethod
    def list_file_set_versions(file_set_name):
        params = {'name': file_set_name}
        return RestRequest(StorageApi.list_file_set_versions) \
            .with_query(params) \
            .with_credentials() \
            .run()

    @staticmethod
    def list_file_sets() -> List[str]:
        return RestRequest(StorageApi.list_file_sets) \
            .with_credentials() \
            .run()
