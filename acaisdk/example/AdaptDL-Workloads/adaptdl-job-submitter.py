import os
import sys
print(os.path.dirname(os.path.realpath('__file__')))
sys.path.append(os.path.dirname(os.path.realpath('__file__')) + '/../../../')
from acaisdk.file import File
from acaisdk.project import Project
from acaisdk.fileset import FileSet
from acaisdk.job import Job
from acaisdk.meta import * 
from acaisdk.utils import utils
from acaisdk import credentials
import argparse
import pandas
import time
# from acaisdk import automl



def setup_workspace():
    os.environ["CLUSTER"] = 'PHOEBE'
    utils.DEBUG = True  # print debug messages
    workspace = os.path.dirname(os.path.realpath('__file__'))  # get current directory
    credentials.login('sWEuAkD8NkLWXHRxhOWcbbesj6Ny1jBx')
    
    # Make your changes here
    project_id = "nravish2_adaptdl_9"
    root_token = 'EmDlCTBF1ppONSciYVd03M9xkmF6hFqW'
    project_admin = 'nravish2'
    user = 'nravish2'
    
    # r = Project.create_project(project_id, root_token, project_admin, csp='AZURE')
    # print(r)
    # print(credentials.login(r['project_admin_token']))


def submit_adaptdl_job(workspace, name, description, input_file_set, application, target_num_replicas, target_batch_size):
    print(f"workspace = {workspace}, name = {name}, description = {description}, \
        input_file_set = {input_file_set}, application = {application}, \
        target_num_replicas = {target_num_replicas}, target_batch_size = {target_batch_size}")
    
    code = os.path.join(workspace, f'{application}.zip')
    File.upload({code: f'/{application}.zip'})
    print(File.list_dir('/'))

    return

    download_dir = workspace + '/temp'
    File.download({f'/{application}.zip' : download_dir})
    # Upload input files and create file set
    input_dir = os.path.join(workspace, 'data')
    File.convert_to_file_mapping([input_dir], 'data/') \
    .files_to_upload \
    .upload() \
    .as_new_file_set(input_file_set)
    
    # Run a job
    job_setting = {
        "v_cpu": "",
        "memory": "",
        "gpu": "",
        "command": "",
        "container_image": "",
        'input_file_set': f'{input_file_set}:1',
        'output_path': '',
        'code': f'/{application}.zip',
        'description': description,
        'name': name,
        'adaptdl_status': 'Enabled',
        'application': application,
        'target_num_replicas': target_num_replicas,
        'target_batch_size': target_batch_size
    }
    j = Job().with_attributes(job_setting).run()
    print(j)



if __name__ == "__main__":
    setup_workspace()
    workload = pandas.read_csv("phoebe_workload.csv")
    start = time.time()
    workspace = os.path.dirname(os.path.realpath('__file__'))  # get current directory
    for row in workload.sort_values(by="time").itertuples():
        # while time.time() - start < row.time:
        #     time.sleep(1)
        description = "random"
        input_file_set = "data"
        print(row)
        #workspace, name, description, input_file_set, application, target_num_replicas, target_batch_size
        submit_adaptdl_job(workspace, row.application, description, 
                           "data", row.application,
                           row.num_replicas, row.batch_size)
        
            














