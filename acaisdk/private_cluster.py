from acaisdk.configs import get_configs
import boto3
import os
import logging
from botocore.exceptions import ClientError

from botocore.client import Config

PRIVATE_CLUSTER = None

def get_private_cluster():
    global PRIVATE_CLUSTER
    if not PRIVATE_CLUSTER:
        PRIVATE_CLUSTER = PrivateCluster()
    return PRIVATE_CLUSTER

class PrivateCluster:

    def __init__(self):
        config = get_configs() 
        self.s3_client = boto3.client('s3', 
            aws_access_key_id=config.access_key_id, 
            aws_secret_access_key=config.secret_access_key, 
            endpoint_url=config.private_cluster_endpoint)
    

    def get_buckets(self):
        return self.s3_client.list_buckets()['Buckets']

    def create_buckets(self, bucket_name):
        return self.s3_client.create_bucket(Bucket=bucket_name)
    
    def upload_file(self, bucket, object_name, file_name):
        object_name = object_name[1:]
        response = self.s3_client.upload_file(file_name, bucket, object_name)
    
    def download_file(self, bucket, object_name, file_name):
        object_name = object_name[1:]
        response = self.s3_client.download_file(bucket, object_name, file_name)
