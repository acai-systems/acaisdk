import os
import configparser

CONFIG = None
CONFIG_FILENAME = 'configs.ini'


def get_configs():
    global CONFIG
    if not CONFIG:
        CONFIG = Configs()
    return CONFIG


class Configs:
    def __init__(self, config_path: str = None) -> None:
        """Load configuration file.

        Called at the beginning of each invocation of acai.py

        :param config_path: config path.
        :return: None
        """
        if not config_path:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       CONFIG_FILENAME)

        conf = configparser.ConfigParser()
        conf.read(config_path)

        self.cred_endpoint = \
            conf.get('general', 'credential_endpoint')
        self.cred_endpoint_port = \
            conf.getint('general', 'credential_endpoint_port')   
        self.private_cred_endpoint = \
             conf.get('phoebe', 'credential_endpoint')
        self.private_cred_endpoint_port = \
            conf.getint('phoebe', 'credential_endpoint_port')   
        self.private_cluster_endpoint = \
            conf.get('storage', 'private_cluster_endpoint')
        self.access_key_id = \
            conf.get('storage', 'access_key_id')
        self.secret_access_key = \
            conf.get('storage', 'secret_access_key')

    def update(self, **kwargs):
        self.cred_endpoint = \
            kwargs.get('cred_endpoint', self.cred_endpoint)
        self.cred_endpoint_port = \
            kwargs.get('cred_endpoint_port', self.cred_endpoint_port)
