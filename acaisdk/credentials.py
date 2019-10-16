import os
from utils.exceptions import *
import configparser

LOCAL_CRED_PATH = os.path.join(os.path.expanduser('~'),
                               '.acai', 'credentials')

CREDENTIALS = None


def get_credentials() -> dict:
    global CREDENTIALS
    if not CREDENTIALS:
        Credentials.load()
    return CREDENTIALS.to_dict()


class Credentials(object):
    def __init__(self):
        global CREDENTIALS
        self.project_id = ''
        self.token = ''
        CREDENTIALS = self

    @staticmethod
    def load():
        """Load credentials (project name and token).

        By default, credentials are saved, like AWS CLI, in user home
        directory.

        Project and token in ENV has higher priority than in file.

        It is required that both project name and token are valid,
        API calls will fail if one of the two fields is empty.

        Usage:
        >>> Credentials.load()
        Or:
        If env is already set, above can be implicitly called by calling any API

        All later API calls will be implicitly using this set of credentials.

        :returns Credentials
        """
        c = Credentials()
        if not c.load_from_env():
            c.load_from_file()
        return c

    def load_from_env(self) -> bool:
        project = os.environ.get('ACAI_PROJECT', None)
        token = os.environ.get('ACAI_TOKEN', None)
        if not project or not token:
            return False
        self.project_id, self.token = project, token
        return True

    def load_from_file(self):
        if not os.path.exists(LOCAL_CRED_PATH):
            _msg = 'Cannot find credential at {}. ' \
                   'Set ACAI_PROJECT and ACAI_TOKEN as ENV variables ' \
                   'or use `acai configure`.'.format(LOCAL_CRED_PATH)
            raise AcaiException(_msg)

        cred_conf = configparser.ConfigParser()
        self.project_id = cred_conf.get('default', 'project_name')
        self.token = cred_conf.get(self.project_id, 'token')

    @staticmethod
    def configure(project_name, token) -> None:
        """Configure a new project in local credentials file

        Notice that ENV variable still has higher priority. i.e. the returned
        object may read from ENV instead of the newly configured credentials.

        Usage:
        >>> Credentials.configure('my_project', '****PROJECT_TOKEN****')
        >>> Credentials.load()  # or implicitly load
        All later API calls will be implicitly using this set of credentials.

        Config file is formatted as:

        [default]
        project_name = test_project

        [test_project]
        token = AJ12398DSD43

        [dummy_project]
        token = DS199DPI3452

        :returns Credentials object
        """
        if not os.path.exists(LOCAL_CRED_PATH):
            # Create a new directory with the new file
            os.makedirs(os.path.dirname(LOCAL_CRED_PATH),
                        mode=0o700, exist_ok=True)
            with open(LOCAL_CRED_PATH, 'w'):
                pass

        cred_conf = configparser.ConfigParser()
        cred_conf.read(LOCAL_CRED_PATH)

        if 'default' not in cred_conf.sections():
            cred_conf.add_section('default')
            cred_conf.set('default', 'project_name', project_name)
        if project_name not in cred_conf.sections():
            cred_conf.add_section(project_name)
        cred_conf.set(project_name, 'token', token)

        with open(LOCAL_CRED_PATH, 'w') as f:
            cred_conf.write(f)

    def to_dict(self):
        # Now only token is needed
        return {'token': self.token}
