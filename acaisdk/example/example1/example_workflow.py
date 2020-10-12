#!/usr/bin/env python
# coding: utf-8

# In[27]:


import os
import sys
sys.path.append(os.path.dirname(os.path.realpath('__file__')) + '/../../../')
from acaisdk.file import File
from acaisdk.project import Project
from acaisdk.fileset import FileSet
from acaisdk.job import Job, JobStatus
from acaisdk.meta import *
from acaisdk.utils import utils
from acaisdk import credentials
# from acaisdk import automl

utils.DEBUG = True  # print debug messages
workspace = os.path.dirname(os.path.realpath('__file__'))  # get current directory


# # Make your changes here
# project_id = "integration_test_dev"
# root_token = 'EmDlCTBF1ppONSciYVd03M9xkmF6hFqW'
# project_admin = 'baljit'
# user = 'baljit'


# # In[35]:


# # @training
# # Create project and user
# r = Project.create_project(project_id, root_token, project_admin)
# # Login is done automatically upon user creation
# r = Project.create_user(project_id, r['project_admin_token'], user)  


# # In[36]:


# # @evaluation
# # you can inspect your token in multiple ways
# print(r['user_token'])
# print(os.environ['ACAI_TOKEN'])
# print(credentials.get_credentials())



credentials.login('QBZQx2IuuJOLQLp8jjL3guVsI50fSng1')

# Upload code
code = os.path.join(workspace, 'wordcount.zip')
File.upload({code: '/wordcount.zip'})

# Upload input files and create file set
input_dir = os.path.join(workspace, 'Shakespeare')
File.convert_to_file_mapping([input_dir], 'Shakespeare/').files_to_upload.upload().as_new_file_set('shakespeare.works')

# Run a job
job_setting = {
    "v_cpu": "0.2",
    "memory": "256Mi",
    "gpu": "0",
    "command": "mkdir -p ./my_output/ && (cat Shakespeare/* | python3 wordcount.py ./my_output/)",
    "container_image": "pytorch/pytorch",
    'input_file_set': 'shakespeare.works',
    'output_path': './my_output/',
    'code': '/wordcount.zip',
    'description': 'count some words from Shakespeare works',
    'name': 'my_acai_job'
}

j = Job().with_attributes(job_setting).run()

# Wait till the job completes
status = j.wait()
if status == JobStatus.FINISHED:
    output_file_set = j.output_file_set
    print("Job done. output file set id:", output_file_set)
else:
    print("Job went wrong:", status)

# Take a look at what's in the output folder
File.list_dir('/my_output')

# Download the result to local device
File.download({'/my_output/wordcount.txt': workspace})
