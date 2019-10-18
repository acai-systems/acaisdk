import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acaisdk.file import File
from acaisdk.project import Project
from acaisdk.utils import utils
from acaisdk.fileset import FileSet
from acaisdk.meta import *

utils.DEBUG = True
utils.IS_CLI = True

# project_id = 'hotpotchicken'
# root_token = 'EmDlCTBF1ppONSciYVd03M9xkmF6hFqW'


# r = Project.create_project('hotpotchicken', root_token)
# print(r)

#{'project_admin_id': 150,
# 'project_admin_token': 'tzWRgulfDgjHBJwEms74gKo8OSphAKYI'}

os.environ.update({'ACAI_PROJECT': 'hotpotchicken',
                   'ACAI_TOKEN': 'tzWRgulfDgjHBJwEms74gKo8OSphAKYI'})
#
# local_path = '/mnt/c/Users/ChangXu/GoogleDrive/CMU.MCDS/Acai/SampleWorkflow2/hotpot_test.tar.gz'
local_path = '/home/xuchang/GoogleDrive/Acai/SampleWorkflow2/bert_eval.py'
remote_path = '/Datui/bert_eval.py'
# r = File.upload({local_path: remote_path}).as_new_file_set('allenhub')
# print(r)

# r = FileSet.download_file_set('allenhub', './')
# r = FileSet.resolve_file_set('allenhub:1')
# c = [Condition('__create_time__').range(0, 1571272230007)]
#
# r = Meta.find_file_set(Condition('__create_time__').range(0, 1571272230007))

r = Meta.get_file_meta(['/Datui/bert_eval.py'])

# r = File.download({'/hotpot_test.tar.gz:8': './res.tar.gz'})
print(r)

# r =