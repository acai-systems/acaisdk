import json
import yaml
from acaisdk.utils.exceptions import *
from collections import OrderedDict
from acaisdk.services.api_calls import *
from acaisdk.fileset import FileSet
from typing import Union, Tuple, List, Dict
from pprint import pformat
import time


class JobStatus(Enum):
    """
    :ivar QUEUEING:
    :ivar LAUNCHING:
    :ivar DOWNLOADING:
    :ivar RUNNING:
    :ivar UPLOADING:
    :ivar FINISHED:
    :ivar FAILED:
    :ivar KILLED:
    :ivar CONTAINER_CRASHED:
    :ivar UNKNOWN: It seems that the job does not exist
    """
    QUEUEING = auto()
    LAUNCHING = auto()
    DOWNLOADING = auto()
    RUNNING = auto()
    UPLOADING = auto()
    FINISHED = auto()
    FAILED = auto()
    KILLED = auto()
    CONTAINER_CRASHED = auto()
    UNKNOWN = auto()

    @staticmethod
    def from_str(string):
        return JobStatus[string.upper().replace(' ', '_')]


class Job:
    """Run a job on the cloud.

    Typical usage (go to `example` folder for a sample workflow
    on Jupyter notebook):

    .. code-block:: python

        command = "mkdir -p ./my_output/ && " \\
                  "(cat Shakespeare/* | python3 wordcount.py ./my_output/)"
        attr = {
            "v_cpu": "0.2",
            "memory": "64Mi",
            "gpu": "0",
            "command": command,
            "container_image": "pytorch/pytorch",
            'input_file_set': 'shakespeare.texts',
            'output_path': './my_output/',
            'code': '/wordcount.zip',
            'description': 'count some words from Shakespeare works',
            'name': 'my_acai_job'
        }

        Job().with_attributes(attr).register().run()


    :ivar id:
        (int) Job id. Not needed for job registration since it does not exist
        at that point (obviously).

    :ivar name: Job name

    :ivar input_file_set:
        This input set will be downloaded to the container where the job is
        executed. If no version given, latest file set is chosen.

    :ivar output_path:
        Only files written to this output folder in the container will be
        uploaded and packed into a new file set. Can be a relative path
        (relative to `/acai/` directory).

    :ivar output_file_set:
        Output file set name. Not need for job submission since the name is
        decided by the backend.

    :ivar code:
        Remote location of the code zip.

    :ivar command:
        The command in which the code is executed. As this is a shell command,
        the execution is carried out by the default shell in the container.
        One benefit is that you can make full use of shell grammars like
        :code:`&&`, :code:`;` and :code:`|`, etc.

    :ivar container_image:
        ID for the docker image so that ACAI can pull the image down to the
        execution host.

    :ivar description:
        Some description for the job. Just like :code:`git commit -m`

    :ivar submitted_time:
        Unix timestamp of the submission time. Not needed for submission.

    :ivar updated_time:
        Deprecated.

    :ivar v_cpu:
        (str) Number of virtual CPUs the execution container can have.
        See doc for `with_resources()` for recommended usage.

    :ivar memory:
        (str) The amount of physical memory for the container.
        Notice that exceeding the limit will result in job failure.
        See doc for `with_resources()` for recommended usage.

    :ivar gpu:
        (str) Number of GPUs to allocate to the container. See doc for
        `with_resources()` for recommended usage.

    """
    __slots__ = [
        'id',
        'name',
        'input_file_set',
        'output_path',
        'output_file_set',
        'code',
        'command',
        'container_image',
        'description',
        'submitted_time',
        'updated_time',
        'v_cpu',
        'gpu',
        'memory',
        'registered',
        'submitted',
        'job_status',
    ]
    _required_fields = [
        'name',
        'input_file_set',
        'output_path',
        'code',
        'command',
        'container_image',
        'description',
        'v_cpu',
        'gpu',
        'memory',
    ]
    _blacklist_fields_print = [

    ]
    _blacklist_fields_submit = [
        'updated_time',
        'registered',
        'submitted'
    ]

    def __init__(self):
        self.registered = False
        self.submitted = False
        self.v_cpu, self.gpu, self.memory = '0.5', '512Mi', '0'
        self.job_status = None

    def register(self):
        """Register the job with ACAI backend. Only registered job can be run.
        """
        if self.registered:
            raise AcaiException('Job already registered')
        self._validate()

        # use full id of input file set
        self.input_file_set = \
            FileSet.list_file_set_content(self.input_file_set)['id']

        data = {k: v for k, v in self.dict.items()
                if k not in self._blacklist_fields_submit}

        r = RestRequest(JobRegistryApi.new_job) \
            .with_data(data) \
            .with_credentials() \
            .run()
        self.with_attributes(r)
        self.registered = True
        debug(r)
        return self

    def run(self) -> 'Job':
        """Execute registered job."""
        if not self.registered:
            raise AcaiException('Job not registered')
        if self.submitted:
            raise AcaiException('Job already submitted')
        r = RestRequest(JobSchedulerApi.new_job) \
            .with_data({'job_id': self.id}) \
            .with_credentials() \
            .run()
        self.submitted = True
        debug(r)
        return self

    def _validate(self):
        fields_not_set = [f for f in self._required_fields
                          if not hasattr(self, f)]
        if fields_not_set:
            _msg = 'Fields not set when submitting job: ' \
                   '{}'.format(fields_not_set)
            raise ArgError(_msg)

    def with_attributes(self, d: dict) -> 'Job':
        """Fill job object with attributes.

        :param d: Dict of attributes to add to the job object.
        :return: Updated job object.
        """
        [setattr(self, k, v) for k, v in d.items() if k in self.__slots__]
        return self

    def with_resources(self,
                       vcpu: Union[int, str] = None,
                       gpu: Union[int, str] = None,
                       mem: Union[int, str] = None) -> 'Job':
        """A more friendly method for adding resource constraints.

        Each of the three parameters can be str or int. For example,

        :code:`mem=1e9` means max mem usage of 1e9 bytes (~1GB)

        :code:`mem="100Mi"` means max mem usage of 100MB

        Memory string format is the same as here:
        https://kubernetes.io/docs/concepts/configuration/manage-compute-resources-container/#meaning-of-memory

        """

        # TODO: autoprovision
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

        if vcpu:
            self.v_cpu = _to_string(vcpu, 'vcpu')
        if gpu:
            self.gpu = _to_string(gpu, 'gpu')
        if mem:
            self.memory = _to_string(mem, 'mem')

        return self

    # ===== INFO =====
    @staticmethod
    def list_jobs_json() -> dict:
        return RestRequest(JobRegistryApi.jobs) \
            .with_credentials() \
            .run()

    @staticmethod
    def list_jobs() -> 'List[Job]':
        """List all jobs under current project.

        :return: a list of Job objects
        """
        return [Job().with_attributes(d) for d in Job.list_jobs_json()]

    @staticmethod
    def find(job_id: int) -> 'Job':
        """Find a job by job ID.

        :param job_id: integer job ID
        :return: Job object of the found job.
        """
        j = Job()
        j.id = job_id
        j.registered = True
        j.with_attributes(j._info())
        return j

    def _info(self):
        return RestRequest(JobRegistryApi.job) \
            .with_query({'job_id': self.id}) \
            .with_credentials() \
            .run()

    # ===== STATUS =====
    @staticmethod
    def check_job_status(job_id) -> JobStatus:
        """Check job status by job ID.

        Usage:

        >>> Job.check_job_status(10)
        """
        return Job.find(job_id).status()

    def status(self) -> JobStatus:
        """Check the status of the current job.

        Usage:

        >>> status = Job.find(10).status()

        In the mean time, output file set is updated. But it will
        only be meaningful when the job is successfully finished. You can
        then access it by

        >>> j = Job.find(123).status()
        >>> output_file_set = j.output_file_set

        :return: :class:`.JobStatus`
        """
        r = RestRequest(JobMonitorApi.job_status) \
            .with_query({'job_id': self.id}) \
            .with_credentials() \
            .run()
        self.output_file_set = r['output_file_set']
        self.job_status = JobStatus.from_str(r['job_status'])
        return self.job_status

    def get_output_file_set(self):
        """Get the output file set of the job.

        Notice that this method is only safe when the job in question is
        submitted and finished. Otherwise it may raise exception.

        The return value only makes sense when the job is successfully
        finished.
        """
        self.status()
        return self.output_file_set

    def wait(self) -> JobStatus:
        """Block until job finish or fail.

        By the way, as wait finishes, the output file set will
        become available.

        example:

        >>> j = Job.with_attributes({...}).register().run()
        >>> if j.wait() == JobStatus.FINISHED:
        >>>     print(j.output_file_set)

        :return: :class:`.JobStatus`
        """
        while 1:
            status = self.status()
            if status in (JobStatus.FINISHED,
                          JobStatus.FAILED,
                          JobStatus.KILLED,
                          JobStatus.CONTAINER_CRASHED,
                          JobStatus.UNKNOWN):
                break
            debug('Current status: {}'.format(status))
            time.sleep(10)
        return status

    @property
    def dict(self) -> OrderedDict:
        """Get a dictionary representation of the job.

        Usage:

        >>> j = Job.find(10)
        >>> d = j.dict

        Just to digress a bit, to get a pretty formatted string
        representation of a job you can just do:

        >>> j = Job.find(10)
        >>> print(j)
        """
        r = OrderedDict()
        [r.update({s: getattr(self, s)}) for s in self.__slots__
         if hasattr(self, s) and s not in self._blacklist_fields_print]
        return r

    @staticmethod
    def from_dict(d: Dict) -> 'Job':
        """Wrapper for :code:`with_attributes()` method.
        Has the exact same behavior.
        """
        return Job().with_attributes(d)

    @staticmethod
    def from_json(path) -> 'Job':
        """Wrapper for :code:`with_attributes()` method. For when you want to
        load the job settings from a JSON file.
        """
        with open(path, 'r') as f:
            return Job.from_dict(json.load(f))

    @staticmethod
    def from_yaml(path) -> 'Job':
        """Wrapper for :code:`with_attributes()` method. For when you want to
        load the job settings from a YAML file.
        """
        with open(path, 'r') as f:
            return Job.from_dict(yaml.load(f))

    def __repr__(self):
        return pformat(dict(self.dict))


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