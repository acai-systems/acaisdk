import os
from acaisdk.utils.exceptions import *
import configparser

LOCAL_CRED_PATH = os.path.join(os.path.expanduser('~'),
                               '.acai', 'credentials')

CREDENTIALS = None


def login(token) -> None:
    """Log in with a new token. ENV variable will be automatically updated.
    :py:meth:`refresh()` is called by default.

    :param token: user token.
    """
    os.environ.update({'ACAI_TOKEN': token})
    refresh()


def refresh() -> None:
    """Refresh credentials. Used when a new token is manually added to ENV.
    """
    Credentials.load()


def get_credentials(force=False) -> dict:
    """Returns credentials as a dictionary for REST requests.

    The function also lazily loads credentials upon invocation.

    User do not need to use this method.
    """
    global CREDENTIALS
    if not CREDENTIALS or force:
        Credentials.load()
    return CREDENTIALS._to_dict()


class Credentials(object):
    """
    Almost all interactions between user and the ACAI backend requires a token
    to identify the user.

    Credentials include a project name (or id, they are the same in ACAI) and
    a token. They are stored inside a configuration file (just like AWS CLI)
    or as environment variables (again, like AWS CLI).

    The easiest way to work with credentials is to store them inside env
    variables, so that you never need to explicitly call any methods in this
    class. All API calls will automatically use methods under this class to
    authenticate with the backend. You just need to do:

    >>> import os
    >>> os.environ['ACAI_PROJECT'] = 'MyAwesomeProject'
    >>> os.environ['ACAI_TOKEN'] = '***************D8S6'
    """

    def __init__(self):
        global CREDENTIALS
        self.project_id = ''
        self.token = ''
        CREDENTIALS = self

    @staticmethod
    def load():
        """Load credentials from ENV or from `~/.acai/credentials`.

        * Credentials from ENV has higher priority than from file.

        * It is required that both project name and token are valid,
          API calls will fail if one of the two fields is empty.

        Usage:

        >>> Credentials.load()

        Or (preferably):
        Just don't call this method. It will be called automatically.

        All later API calls will be implicitly using this set of credentials.

        :return: Credentials object
        """
        c = Credentials()
        if not c._load_from_env():
            c._load_from_file()
        return c

    def _load_from_env(self) -> bool:
        # project = os.environ.get('ACAI_PROJECT', None)
        project = '__dummy_value__'  # TODO: Now we don't need project
        token = os.environ.get('ACAI_TOKEN', None)
        if not project or not token:
            return False
        self.project_id, self.token = project, token
        return True

    def _load_from_file(self) -> None:
        if not os.path.exists(LOCAL_CRED_PATH):
            _msg = 'Cannot find credential at {}. ' \
                   'Set ACAI_PROJECT and ACAI_TOKEN as ENV variables ' \
                   'or use `acai configure`.'.format(LOCAL_CRED_PATH)
            raise AcaiException(_msg)

        cred_conf = configparser.ConfigParser()
        cred_conf.read(LOCAL_CRED_PATH)
        self.project_id = cred_conf.get('default', 'project')
        self.token = cred_conf.get(self.project_id, 'token')

    @staticmethod
    def configure(project_name, token) -> None:
        """Configure a new project in local credentials file

        Notice that ENV variable still has higher priority even if you use this
        method to store a new set of credentials to file.

        Usage:

        >>> Credentials.configure('my_project', '****PROJECT_TOKEN****')
        >>> Credentials.load()  # or implicitly load

        Credential file is formatted as:

        .. code-block:: text

            [default]
            project_name = test_project

            [test_project]
            token = ************SD43

            [dummy_project]
            token = ************3452

        Note: Writing to credential file is not tested. Use ENV to authenticate.

        :return: Credentials object
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

    def _to_dict(self):
        """Returns credentials as a dictionary for REST requests."""
        return {'token': self.token}
