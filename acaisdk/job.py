import json
import yaml
from acaisdk.utils.exceptions import *
from acaisdk.services.api_calls import *


class Job:
    __slots__ = [
        'job_id',
        'project_id',
        'user_id',
        'input_fileset_id',
        'output_path',
        'code',
        'command',
        'container_image',
        'name',
        'description',
        'time_submitted',
        'resource',
        'registered',
        'submitted']
    required_fields = [
        'project_id',
        'user_id',
        'input_fileset_id',
        'output_path',
        'code',
        'command',
        'container_image',
        'name',
        'description',
        'resource']
    blacklist_fields = [
        'registered',
        'submitted'
    ]

    def __init__(self):
        self.resource = {}
        self.registered = False
        self.submitted = False

    def register(self):
        if self.registered:
            raise AcaiException('Job already registered')
        self._validate()
        r = RestRequest(JobManager.submit_job) \
            .with_data(self.dict) \
            .with_credentials() \
            .run()
        self.with_attributes(r)
        self.registered = True
        return self

    def run(self):
        if not self.registered:
            raise AcaiException('Job not registered')
        if self.submitted:
            raise AcaiException('Job already submitted')
        # TODO
        self.submitted = True

    def info(self):
        r = RestRequest(JobManager.job_info) \
            .with_query({'job_id': self.job_id}) \
            .with_credentials() \
            .run()
        self.with_attributes(r)

    @staticmethod
    def list_jobs():
        """User id is read implicitly"""
        pass

    # ==== Monitor ===
    def update_job_status(self):
        pass  # not open for CLI

    def status(self):
        r = RestRequest(JobManager.job_status) \
            .with_query({'job_id': self.job_id}) \
            .with_credentials(g) \
            .run()
        return r

    def _validate(self):
        fields_not_set = [f for f in self.required_fields
                          if not hasattr(self, f)]
        if fields_not_set:
            _msg = 'Fields not set when submitting job: ' \
                   '{}'.format(fields_not_set)
            raise ArgError(_msg)

    def with_attributes(self, d: dict):
        [setattr(self, k, v) for k, v in d.items()]
        return self

    def with_resource(self, vcpu=None, gpu=None, mem=None):
        def _to_string(v, name):
            if type(v) == int:
                return '{}'.format(v)
            elif type(v) == tuple:
                if len(v) == 1:
                    return '{}'.format(v[0])
                elif len(v) == 2:
                    return '{}-{}'.format(*v)
            _msg = 'Wrong type of argument for resource {}: {}'.format(name, v)
            raise ArgError(_msg)

        result = {}
        if vcpu:
            result['vcpu'] = _to_string(vcpu, 'vcpu')
        if gpu:
            result['gpu'] = _to_string(gpu, 'gpu')
        if mem:
            result['mem'] = _to_string(mem, 'mem')

        self.resource.update(result)
        return self

    @property
    def dict(self):
        return {s: getattr(self, s) for s in self.__slots__
                if hasattr(self, s) and s not in self.blacklist_fields}

    @staticmethod
    def from_dict(d: dict):
        return Job().with_attributes(d)

    @staticmethod
    def from_json(path):
        with open(path, 'r') as f:
            return Job.from_dict(json.load(f))

    @staticmethod
    def from_yaml(path):
        with open(path, 'r') as f:
            return Job.from_dict(yaml.load(f))


class ProfilingJob(object):
    __slots__ = [
        'id',
        'profiling_config',
        'input_fileset_id',
        'command']

    def __init__(self, job: Job, input_fileset_id: int, command: str):
        self.profiling_config = job
        self.input_fileset_id = input_fileset_id
        self.command = command

    @property
    def dict(self):
        def _object_to_dict(o):
            return o.dict if type(o) == Job else o

        return {s: _object_to_dict(getattr(self, s))
                for s in self.__slots__ if hasattr(self, s)}
