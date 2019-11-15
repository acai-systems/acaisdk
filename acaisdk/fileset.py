from acaisdk.services.api_calls import *
from typing import Iterable, List
from acaisdk import file


class FilesList(list):
    def as_new_file_set(self, file_set_name):
        """Create a file set for a newly uploaded batch.

        Usage can be found at `File.upload` and `File.convert_to_file_mapping`
        """
        return FileSet.create_file_set(file_set_name,
                                       [_[1] for _ in self])

    def upload(self):
        """Upload a FilesList to data lake.

        Usage can be found at `File.convert_to_file_mapping`
        """
        return file.File.upload(self)


class FileSet:
    @staticmethod
    def create_file_set(file_set_name: str,
                        remote_entities: list) -> dict:
        """Create a file set on a list of remote files or file sets.

        Denoting a file is the same as anywhere else. Use "@" prefix to
        denote file sets.

        Examples:

        1. Create file set from files

        .. code-block::

            create_file_set("my_new_file_set_name",
                            ["/my_data/test.json", "/my_data/a/b.txt:3"])

        2. Create file set from other file sets

        .. code-block::

            create_file_set("my_new_file_set_name",
                            ["@file_set_a:1", "@file_set_b"])

        3. You can also mix file and file sets

        .. code-block::

            create_file_set("my_new_file_set_name",
                            ["@file_set_a:1",
                             "/my_data/a/b.txt:3",
                             "@file_set_c"]
                            )


        :return:

            .. code-block:: text

                {
                  "id": "HotpotQA:1",
                  "files": [
                    "data/train.json:2"
                  ]
                }
        """
        if ':' in file_set_name:
            _msg = 'Do not create file set with ":" in its name.'
            raise AcaiException(_msg)
        if type(remote_entities) not in (tuple, list):
            _msg = 'Need a list or tuple of remote entities ' \
                   'instead of {}.'.format(type(remote_entities))
            raise AcaiException(_msg)
        data = {
            "name": file_set_name,
            "files": list(remote_entities)
        }
        return RestRequest(StorageApi.create_file_set) \
            .with_data(data) \
            .with_credentials() \
            .run()

    @staticmethod
    def list_file_set_content(vague_name) -> dict:
        """List all files in a file set."""
        params = {'vague_name': vague_name}
        return RestRequest(StorageApi.resolve_file_set) \
            .with_query(params) \
            .with_credentials() \
            .run()

    @staticmethod
    def resolve_vague_name(vague_name):
        """Under construction, don't use this"""
        return file.File.resolve_vague_path(vague_name)

    @staticmethod
    def download_file_set(vague_name: str,
                          mount_point: str = None,
                          force: bool = False) -> None:
        """Download a file set to local device.

        :param vague_name:
            File set name can be vague (with or without version number). Latest
            version of the file set will be chosen if no version given.

        :param mount_point:
            Which local directory to download the file set to. This won't
            actually "mount" any device on your local file system. But the
            behavior will be similar to mounting the root directory in the
            remote file system to the "mount_point" directory.

            e.g. For a file set `allen:2` with file

            .. code-block:: text

                /allen/1.txt:1
                /allen/d/2.txt:1
                /allen/3.txt:3

            calling :code:`download_file_set('allen:2', '/local_tmp/')`
            results in a local directory hierarchy of:

            .. code-block:: text

                /local_tmp/allen/1.txt
                /local_tmp/allen/d/2.txt
                /local_tmp/allen/3.txt

            Notice that the SDK won't create the mount_point directory for you,
            but it will create folders inside the mount_point automatically.

        :param force:
            If local directory has conflicting file names, choose if
            continue to download.

        """
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
        """Show all file sets under the project."""
        return RestRequest(StorageApi.list_file_sets) \
            .with_credentials() \
            .run()
