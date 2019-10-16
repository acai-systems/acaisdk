from credentials import *
from services.api_calls import *
from utils.fileops import FileIO
from typing import Dict


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
            .with_credentials(get_credentials()) \
            .run()
        return r

    @staticmethod
    def upload(local_to_remote_mapping: Dict[str, str]) -> Dict[str, str]:
        """Upload multiple files. Duplicated upload is not dealt with.

        :return: {local_path: remote_path:version}
        """
        versioned_mapping = {}
        for local_path, remote_path in local_to_remote_mapping.items():
            r = File._get_upload_link(remote_path)
            s3_url = r['s3_url']
            file_id = r['id']
            FileIO(local_path).upload(s3_url)
            versioned_mapping[local_path] = file_id
        return versioned_mapping

    @staticmethod
    def download(remote_to_local_mapping: Dict[str, str]) -> None:
        for remote_path, local_path in remote_to_local_mapping.items():
            s3_url = File._get_download_link(remote_path)['s3_url']
            FileIO.download(s3_url, local_path)

    @staticmethod
    def _get_upload_link(remote_path):
        """
        :return:    {
                       "id": "data/train.json:2",
                       "s3_url": "url"
                     }
        """
        return RestRequest(Storage.upload_file) \
            .with_data({'path': remote_path}) \
            .with_credentials(get_credentials()) \
            .run()

    @staticmethod
    def _get_download_link(remote_path):
        return RestRequest(Storage.download_file) \
            .with_query({'path': remote_path}) \
            .with_credentials(get_credentials()) \
            .run()
