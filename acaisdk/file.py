from acaisdk.services.api_calls import *
from acaisdk.utils.fileops import FileIO
from typing import Dict, List, Tuple, Union
from acaisdk import fileset
from acaisdk.utils import utils
import os
import glob


class File:
    @staticmethod
    def list_dir(directory: str):
        """
        :return:    [
                      {
                        "name": "train.json",
                        "version": 1,
                        "is_dir": false
                      }
                    ]
        """
        r = RestRequest(Storage.list_directory) \
            .with_query({'directory_path': directory}) \
            .with_credentials() \
            .run()
        return r

    @staticmethod
    def upload(local_to_remote: Union[Dict[str, str],
                                      List[Tuple[str, str]]],
               results: list = None):
        """Upload multiple files. Duplicated upload is not dealt with.

        :param local_to_remote: {local_path: remote_path, ...}
                                or [(local_path, remote_path), ...]
        :param results: Store result to another FilesList, for when you are
                        interested in the result but want to chain this method
                        with other methods like:
                        >>> File.upload([('/a', '/b')], []).as_new_file_set()
        :return: fileset.FilesList
        """
        l_r_mapping = local_to_remote
        if type(local_to_remote) == dict:
            l_r_mapping = local_to_remote.items()

        versioned_mapping = fileset.FilesList()
        for local_path, remote_path in l_r_mapping:
            r = File._get_upload_link(remote_path)
            s3_url, file_id = r['s3_url'], r['id']
            if utils.IS_CLI:
                print('Uploading file {} to {}'
                      ''.format(local_path, remote_path))
            FileIO(local_path).upload(s3_url)
            versioned_mapping.append((local_path, file_id))

        if results:
            results += versioned_mapping
        return versioned_mapping

    @staticmethod
    def download(remote_to_local: Dict[str, str]) -> None:
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
        return RestRequest(Storage.upload_file) \
            .with_data({'path': remote_path}) \
            .with_credentials() \
            .run()

    @staticmethod
    def _get_download_link(remote_path):
        return RestRequest(Storage.download_file) \
            .with_query({'path': remote_path}) \
            .with_credentials() \
            .run()

    @staticmethod
    def convert_to_file_mapping(local_paths: List[str], remote_path):
        """The method is not atomic"""
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
                r = os.path.join(remote_path,
                                 os.path.basename(path))
                l_r_mapping.append((path, r))
            else:
                all_ignores.append(path)

        return l_r_mapping, all_ignores

    @staticmethod
    def _is_dir(path: str):
        if path.endswith('/'):
            return True
        if os.path.exists(path) and os.path.isdir(path):
            return True

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
