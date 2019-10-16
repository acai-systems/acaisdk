import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file import File
from project import Project
from utils import utils

utils.DEBUG = True
utils.IS_CLI = True

project_id = 'hotpotchicken'
root_token = 'EmDlCTBF1ppONSciYVd03M9xkmF6hFqW'


# r = Project.create_project('hotpotchicken', root_token)
# print(r)

#{'project_admin_id': 150,
# 'project_admin_token': 'tzWRgulfDgjHBJwEms74gKo8OSphAKYI'}

os.environ.update({'ACAI_PROJECT': 'hotpotchicken',
                   'ACAI_TOKEN': 'tzWRgulfDgjHBJwEms74gKo8OSphAKYI'})
#
# local_path = '/mnt/c/Users/ChangXu/GoogleDrive/CMU.MCDS/Acai/SampleWorkflow2/hotpot_test.tar.gz'
# remote_path = '/hotpot_test.tar.gz'
# r = File.upload({local_path: remote_path})
# print(r)

r = File.download({'/hotpot_test.tar.gz:8': './res.tar.gz'})
print(r)
