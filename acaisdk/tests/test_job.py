import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acaisdk.file import File
from acaisdk.project import Project
from acaisdk.utils import utils
from acaisdk.fileset import FileSet
from acaisdk.job import Job, JobStatus
from pprint import pprint

utils.DEBUG = True
utils.IS_CLI = True

os.environ.update({'ACAI_TOKEN': 'LqMcGvgN9LlxvAGoyniFYapU6btKJY9b'})

# Job().with_attributes({
#     "v_cpu": "0.5",
#     "memory": "320Mi",
#     "gpu": "0",
#     "command": "echo hello world from default job",
#     "container_image": "pytorch",
#     'input_file_set': 'sterling',
#     'output_path': 'outputlalapath',
#     'code': '/albertinputs/demo.zip',
#     'description': 'nothinghere',
#     'name': 'rubbish2'
# }).register().run()

# pprint(Job.list_jobs_json()[-1])
# print(Job.check_job_status(50))
print(JobStatus['container crashed'.upper().replace(' ', '_')].name)