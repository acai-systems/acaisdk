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
            config_path = os.path.join(os.path.dirname(__file__),
                                       CONFIG_FILENAME)

        conf = configparser.ConfigParser()
        conf.read(config_path)

        self.cred_endpoint = \
            conf.get('general', 'credential_endpoint')
        self.cred_endpoint_port = \
            conf.getint('general', 'credential_endpoint_port')

    def update(self, **kwargs):
        self.cred_endpoint = \
            kwargs.get('cred_endpoint', self.cred_endpoint)
        self.cred_endpoint_port = \
            kwargs.get('cred_endpoint_port', self.cred_endpoint_port)
