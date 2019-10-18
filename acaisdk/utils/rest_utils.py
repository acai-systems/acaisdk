import urllib3
import requests
from http import HTTPStatus
from acaisdk.utils import exceptions

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SESSION = None


def get_session() -> requests.Session:
    global SESSION
    if not SESSION:
        SESSION = requests.Session()
    return SESSION


def post(server, port, service, path,
         params: dict, data: dict):
    r = get_session().post('https://{}:{}/{}/{}'
                           .format(server, port, service, path),
                           params=params,
                           json=data,
                           verify=False)
    if r.status_code != HTTPStatus.OK:
        raise exceptions.RemoteException(r.content)
    return r.json()


def get(server, port, service, path, params: dict):
    r = get_session().get('https://{}:{}/{}/{}'
                          .format(server, port, service, path),
                          params=params, verify=False)
    if r.status_code != HTTPStatus.OK:
        raise exceptions.RemoteException(r.content)
    return r.json()
