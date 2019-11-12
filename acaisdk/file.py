from acaisdk.services.api_calls import *
from acaisdk.utils.fileops import FileIO
from typing import Dict, List, Tuple, Union, NamedTuple
from acaisdk import fileset
from acaisdk.utils import utils
import os
import glob
import time
from collections import namedtuple


class File:
    @staticmethod
    def list_dir(directory: str) -> List[Dict]:
        """List all files and directories in a remote directory.

        "version" denotes the latest version of a file.
        Notice that version number for directories makes no sense.

        :return:

            .. code-block::

                [
                    {
                    "path": "train.json",
                    "version": 1,
                    "is_dir": false
                    },
                    ...
                ]
        """
        r = RestRequest(StorageApi.list_directory) \
            .with_query({'directory_path': directory}) \
            .with_credentials() \
            .run()
        return r

    @staticmethod
    def upload(local_to_remote: Union[Dict[str, str],
                                      List[Tuple[str, str]]],
               results: list = None) -> 'fileset.FilesList':
        """Upload multiple files.

        Notice that the method does not deal with conflicting updates. It is
        up to the user to make sure there is no unintended uploads to the same
        remote location. Otherwise, multiple versions of the same file path
        will be created. (i.e. you won't lose any data)

        :param local_to_remote: Allows dictionary or list of tuples. E.g.

                :code:`{local_path: remote_path, ...}` or
                :code:`[(local_path, remote_path), ...]`

        :param results: Store result to another FilesList, for when you are
                        interested in the result but want to chain this method
                        with other methods like:

                        >>> File.upload([('/a', '/b')], []).as_new_file_set()

        :return:
            FilesList object. It's just a python list with additional
            functions for file and file set operations.

            .. code-block::

                [("local_path", "remote_path:version"), ...]
        """
        # No matter the format, convert to List[(local, remote), ...]
        l_r_mapping = local_to_remote
        if type(local_to_remote) == dict:
            l_r_mapping = local_to_remote.items()
        l_r_mapping = list(l_r_mapping)  # make sure it is ordered

        # Get URLs
        remote_paths = [r for _, r in l_r_mapping]
        r = RestRequest(StorageApi.start_file_upload_session) \
            .with_data({'paths': remote_paths}) \
            .with_credentials() \
            .run()
        session_id = r['session_id']

        debug(l_r_mapping)
        for i, (local_path, remote_path) in enumerate(l_r_mapping):
            s3_url = r['files'][i]['s3_url']
            FileIO(local_path).upload(s3_url)
            print('Uploaded {} to {}'.format(local_path, remote_path))

        while 1:
            r = RestRequest(StorageApi.poll_file_upload_session) \
                .with_query({'session_id': session_id}) \
                .with_credentials() \
                .run()
            if r['committed']:
                versioned_remote_paths = r['uploaded_file_ids']
                break
            time.sleep(1)

        # Finish session
        r = RestRequest(StorageApi.finish_file_upload_session) \
            .with_data({'session_id': session_id}) \
            .with_credentials() \
            .run()

        versioned_mapping = fileset.FilesList(
            [(l, vr) for (l, _), vr in zip(l_r_mapping, versioned_remote_paths)]
        )

        if results:
            results += versioned_mapping

        return versioned_mapping

    @staticmethod
    def download(remote_to_local: Dict[str, str]) -> None:
        """ Download multiple remote files to local.

        If version is not specified for remote file, then the latest version
        will be downloaded by default.

        Local path can be a directory, e.g. For a input dict
        :code:`{"/my_acai/b/c.json": "/home/ubuntu/stuff/"}`, `c.json` will be
        downloaded to `/home/ubuntu/stuff/c.json`

        :param remote_to_local:
            :code:`{"remote_path:version": "local_path", ...}`

        :return: None
        """
        for remote_path, local_path in remote_to_local.items():
            s3_url = File._get_download_link(remote_path)['s3_url']
            if os.path.isdir(local_path):
                local_path = os.path.join(local_path,
                                          os.path.basename(remote_path))
            if utils.IS_CLI:
                print('Downloading file {} to {}'
                      ''.format(remote_path, local_path))
            FileIO.download(s3_url, local_path)

    @staticmethod
    def _get_upload_link(remote_path):
        return RestRequest(StorageApi.upload_file) \
            .with_data({'path': remote_path}) \
            .with_credentials() \
            .run()

    @staticmethod
    def _get_download_link(remote_path):
        return RestRequest(StorageApi.download_file) \
            .with_query({'path': remote_path}) \
            .with_credentials() \
            .run()

    class UploadFileMapping(NamedTuple):
        """
        Field number 0: A :class:`.fileset.FilesList` for successfully
        mapped paths.

        Field number 1: A list of files that are not accessible.
        (no permission, maybe).
        """
        files_to_upload: 'fileset.FilesList'
        files_ignored: List[str]

    @staticmethod
    def convert_to_file_mapping(local_paths: List[str],
                                remote_path: str,
                                ignored_paths: List[str] = None
                                ) -> UploadFileMapping:
        """A nice method to make you happy.

        Converts local file and directory paths to their
        corresponding remote paths. So that you do not need to specify
        local to remote path mappings one by one for the `upload` function.

        For a local file system like

        .. code-block:: text

            /a/b/c/1.txt
            /a/b/c/d/2.txt
            /a/b/3.txt

        Running

        .. code-block::

            convert_to_file_mapping(['/a/b/3.txt', '/a/b/c/'], '/allen/')

        will result in a remote file system structure like:

        .. code-block:: text

            /allen/1.txt
            /allen/d/2.txt
            /allen/3.txt

        Notice that if you are writing to a remote directory, a `"/"` must be
        added at the end of the path string, like `"/allen/"` instead of
        `"/allen"`.

        Example usage:

        .. code-block:: python

            File.convert_to_file_mapping(['/a/b/c/'], '/allen/') \\
                .files_to_upload \\
                .upload() \\
                .as_new_file_set('my_training_files')

        Notice that the method is not transactional. It does not protect
        itself from change of files in local directories.

        :return:
            :class:`.File.UploadFileMapping`
        """
        if type(local_paths) != list:
            raise AcaiException('local_paths must be a list!!!')

        l_r_mapping = fileset.FilesList()
        all_ignores = []

        if File._is_dir(remote_path):
            # write to a remote folder
            for path in local_paths:
                if File._is_dir(path):
                    to_upload, to_ignore = File._list_all_files(path)
                    all_ignores += to_ignore
                    for l in to_upload:
                        r = os.path.join(remote_path,
                                         os.path.relpath(l, path))
                        l_r_mapping.append((l, r))
                else:
                    if os.access(path, os.R_OK):
                        r = os.path.join(remote_path,
                                         os.path.basename(path))
                        l_r_mapping.append((path, r))
                    else:
                        all_ignores.append(path)
        else:
            # 1 to 1 file upload
            if len(local_paths) > 1:
                raise AcaiException('Cannot upload multiple files to'
                                    'the same remote file path.')
            if File._is_dir(local_paths[0]):
                raise AcaiException('Cannot upload a local directory to '
                                    'a remote file.')

            path = local_paths[0]
            if os.access(path, os.R_OK):
                r = os.path.join(remote_path)
                l_r_mapping.append((path, r))
            else:
                all_ignores.append(path)

        if ignored_paths is not None:
            ignored_paths += all_ignores

        return File.UploadFileMapping(l_r_mapping, all_ignores)

    @staticmethod
    def _is_dir(path: str):
        if path.endswith('/'):
            return True
        if os.path.exists(path) and os.path.isdir(path):
            return True
        return False

    @staticmethod
    def _list_all_files(dir_path):
        to_upload = []
        inaccessible = []
        for f in glob.glob(dir_path + "/**/*", recursive=True):
            if os.access(f, os.R_OK):
                if not os.path.isdir(f):
                    to_upload.append(f)
            else:
                inaccessible.append(f)
        return to_upload, inaccessible

    @staticmethod
    def list_file_versions(file_name):
        params = {'path': file_name}
        return RestRequest(StorageApi.list_file_versions) \
            .with_query(params) \
            .with_credentials() \
            .run()

    @staticmethod
    def resolve_vague_path(vague_path):
        params = {'vague_path': vague_path}
        return RestRequest(StorageApi.resolve_vague_path) \
            .with_query(params) \
            .with_credentials() \
            .run()
