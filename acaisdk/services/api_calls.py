from enum import Enum, auto
import configs
from utils import rest_utils
from utils.utils import *
from utils.exceptions import *
from collections import namedtuple

E = namedtuple('E', ['id', 'method'])


class RestMethods(Enum):
    get = auto()
    post = auto()


class EnumFactory(Enum):
    @classmethod
    def GET(cls):
        return E(auto(), RestMethods.get)

    @classmethod
    def POST(cls):
        return E(auto(), RestMethods.post)


class Services(Enum):
    @property
    def service_name(self):
        return {Credential: 'credential',
                Storage: 'storage',
                Metadata: 'meta',
                JobManager: 'job_manager',
                Provenance: 'provenance'}[type(self)]

    @property
    def endpoint(self):
        """There is only one endpoint the client talks to."""
        conf = configs.get_configs()
        return conf.cred_endpoint, conf.cred_endpoint_port

    @property
    def method(self):
        return self.value.method


class Credential(Services):
    create_project = EnumFactory.POST()
    create_user = EnumFactory.POST()
    resolve_user_id = EnumFactory.GET()
    resolve_project_id = EnumFactory.GET()


class Storage(Services):
    list_directory = EnumFactory.GET()
    upload_file = EnumFactory.POST()
    download_file = EnumFactory.GET()
    resolve_file_id = EnumFactory.GET()
    create_file_set = EnumFactory.POST()
    resolve_vague_path = EnumFactory.POST()


class JobManager(Services):
    submit_job = EnumFactory.POST()
    job_info = EnumFactory.GET()
    list_jobs = EnumFactory.GET()
    new_job = EnumFactory.GET()
    kill_job = EnumFactory.GET()
    update_job_status = EnumFactory.POST()
    job_status = EnumFactory.GET()
    submit_profiling_job = EnumFactory.POST()
    estimate = EnumFactory.GET()
    autoprovision = EnumFactory.POST()


class Metadata(Services):
    # Project init
    init_project = EnumFactory.POST()

    # File metadata
    init_file_meta = EnumFactory.POST()
    update_file_meta = EnumFactory.POST()
    get_file_meta = EnumFactory.POST()
    del_file_meta = EnumFactory.POST()

    # File set metadata
    init_file_set_meta = EnumFactory.POST()
    update_file_set_meta = EnumFactory.POST()
    get_file_set_meta = EnumFactory.POST()
    del_file_set_meta = EnumFactory.POST()

    # Job Metadata
    init_job_meta = EnumFactory.POST()
    update_job_meta = EnumFactory.POST()
    get_job_meta = EnumFactory.POST()
    del_job_meta = EnumFactory.POST()

    # Query
    query_meta = EnumFactory.POST()


class Provenance(Services):
    register = EnumFactory.POST()


class RestRequest:
    def __init__(self, service: Services):
        self.service = service
        self.query = {}
        self.data = {}
        self.credentials = {}

    def with_query(self, query: dict):
        self.query = query
        return self

    def with_data(self, data: dict):
        self.data = data
        return self

    def with_credentials(self, credentials: dict):
        self.credentials = credentials
        return self

    def run(self):
        endpoint, port = self.service.endpoint
        if self.service.method == RestMethods.get:
            self.query.update(self.credentials)
            debug('Running GET request:',
                  endpoint,
                  port,
                  self.service.service_name,
                  self.service.name)
            return rest_utils.get(endpoint,
                                  port,
                                  self.service.service_name,
                                  self.service.name,
                                  self.query)
        elif self.service.method == RestMethods.post:
            self.data.update(self.credentials)
            debug('Running POST request:',
                  endpoint,
                  port,
                  self.service.service_name,
                  self.service.name)
            debug(self.data)
            return rest_utils.post(endpoint,
                                   port,
                                   self.service.service_name,
                                   self.service.name,
                                   self.query,
                                   self.data)
        raise AcaiException('Unknown request type: '
                            '{}'.format(self.service.method))
