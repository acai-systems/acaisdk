from acaisdk.file import File
from acaisdk.project import Project
from acaisdk.fileset import FileSet
from acaisdk.job import Job
from acaisdk.meta import *

project_id = 'shakespeare'
root_token = 'EmDlCTBF1ppONSciYVd03M9xkmF6hFqW'
project_admin = 'proj_admin'
user = 'ATeam'

workspace = os.path.dirname(__file__)

# Create project and user
r = Project.create_project(project_id, root_token, 'proj_admin')
r = Project.create_user(project_id, r['project_admin_token'], user)
os.environ.update({'ACAI_TOKEN': r['user_token']})

# Upload code
code = os.path.join(workspace, 'wordcount.zip')
File.upload({code: '/wordcount.zip'})

# Upload input files and create file set
input_dir = os.path.join(workspace, 'Shakespeare')
File.convert_to_file_mapping([input_dir], 'Shakespeare/')[0]\
    .upload()\
    .as_new_file_set('shakespeare.works')

# Run a job
job_setting = {
    "v_cpu": "0.5",
    "memory": "320Mi",
    "gpu": "0",
    "command": "cat Shakespeare/* | python3 wordcount.py ./my_output/",
    "container_image": "pytorch/pytorch",
    'input_file_set': 'shakespeare.works',
    'output_path': './my_output/',
    'code': '/wordcount.zip',
    'description': 'count some words',
    'name': 'test job submission'
}

j = Job().with_attributes(job_setting).register().run()

# Check result

print(r)