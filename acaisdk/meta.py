from acaisdk.services.api_calls import *
from typing import Iterable, List
from acaisdk import fileset
from acaisdk import file


class ConditionType(Enum):
    MIN = auto()
    MAX = auto()
    RANGE = auto()
    ARRAY = auto()
    VALUE = auto()

    def __str__(self):
        return self.name.lower()


class Condition:
    """Constrains to apply when filtering out jobs, files and
    file sets.

    Example: find the job output of the best performing CNN model by
    applying multiple Conditions.

    .. code-block:: python

        Meta.find_file_set(
            Condition('model').value('cnn'),
            Condition('accuracy').max(),
            )

    Notice that conditions are applied in an "AND" fashion.
    "OR" logic is not supported.
    """

    def __init__(self, key: str):
        self.key = key
        self.val = None
        self.type = None

    def value(self, val) -> 'Condition':
        """Find entity with value equals :code:`val`
        """
        self.type = ConditionType.VALUE
        self.val = val
        return self

    def max(self) -> 'Condition':
        """Find entity with maximum value
        """
        self.type = ConditionType.MAX
        return self

    def min(self) -> 'Condition':
        """Find entity with minimum value
        """
        self.type = ConditionType.MIN
        return self

    def range(self, start, end) -> 'Condition':
        """Find entity with value ranging from :code:`start` (exclusive)
         to :code:`end` (inclusive)
        """
        self.type = ConditionType.RANGE
        self.val = [start, end]
        return self

    def array(self, array: Iterable) -> 'Condition':
        """Find entity with values matching every value in the input array.
        Which implies that the metadata value must be of array type.
        """
        self.type = ConditionType.ARRAY
        self.val = list(array)
        return self

    def to_dict(self):
        d = {
            "key": self.key,
            "type": str(self.type)
        }
        if self.val:
            d['value'] = self.val
        return d

    def __repr__(self):
        return '{{type: {}, key: {}, val: {}}}'.format(
            self.type, self.key, self.val)


class Meta:
    # === UPDATE ===
    @staticmethod
    def update_file_meta(file_path,
                         tags: list = None,
                         kv_pairs: dict = None):
        """Add new meta data to a file.

        :param file_path: can be without version. Latest is used by default.
        :param tags: a list of tags, same as adding a "tags" key in kv_pairs
        :param kv_pairs: key-value pairs of metadata.
        """
        # Resolve the potentially vague path to explicit path
        r = file.File.resolve_vague_path(file_path)

        if r['is_dir']:
            raise AcaiException('Remote path {} is a directory.'
                                'Directories do not have metadata'
                                ''.format(r['id']))
        explicit_path = r['id']

        tags = [] if not tags else tags
        kv_pairs = {} if not kv_pairs else kv_pairs
        kv_pairs['tags'] = kv_pairs.get('tags', tags)

        data = {'id': explicit_path,
                'meta': kv_pairs}

        return RestRequest(MetadataApi.update_file_meta) \
            .with_data(data) \
            .with_credentials() \
            .run()

    @staticmethod
    def update_file_set_meta(file_set,
                             tags: list = None,
                             kv_pairs: dict = None):
        """Same usage as :py:meth:`~Meta.update_file_meta`"""
        # Resolve the potentially vague name to explicit name
        r = fileset.FileSet.resolve_vague_name(file_set)

        explicit_name = r['id']

        tags = [] if not tags else tags
        kv_pairs = {} if not kv_pairs else kv_pairs
        kv_pairs['tags'] = kv_pairs.get('tags', tags)

        data = {'id': explicit_name,
                'meta': kv_pairs}

        return RestRequest(MetadataApi.update_file_set_meta) \
            .with_data(data) \
            .with_credentials() \
            .run()

    @staticmethod
    def update_job_meta(job_id,
                        tags: list = None,
                        kv_pairs: dict = None):
        """Same usage as :py:meth:`~Meta.update_file_meta`"""
        tags = [] if not tags else tags
        kv_pairs = {} if not kv_pairs else kv_pairs
        kv_pairs['tags'] = kv_pairs.get('tags', tags)

        data = {'id': job_id,
                'meta': kv_pairs}

        return RestRequest(MetadataApi.update_job_meta) \
            .with_data(data) \
            .with_credentials() \
            .run()

    @staticmethod
    def _merge_dict(dict_a, dict_b):
        merged = {}
        if dict_a:
            merged.update(dict_a)
        if dict_b:
            merged.update(dict_b)
        return merged

    # === SEARCHING ===
    @staticmethod
    def find_file(*conditions: Condition):
        """File a job that meets a list of constraints.


        """
        return Meta.query_meta('file', *conditions)

    @staticmethod
    def find_job(*conditions: Condition):
        """Find a job that meets a list of constraints.
        Same usage as :py:meth:`~Meta.find_file`
        """
        return Meta.query_meta('job', *conditions)

    @staticmethod
    def find_file_set(*conditions: Condition):
        """Find a file set that meets a list of constraints.
        Same usage as :py:meth:`~Meta.find_file`
        """
        return Meta.query_meta('fileset', *conditions)

    @staticmethod
    def query_meta(entity_type: str, *conditions: Condition):
        """Base method for querying files, jobs and file sets.

        It is recommended to use :py:meth:`~Meta.find_file`,
        :py:meth:`~Meta.find_job` and :py:meth:`~Meta.find_file_set` instead.
        """
        if entity_type.lower() not in ('file', 'job', 'fileset'):
            raise AcaiException('Entity type must be one of: '
                                '("File", "Job", "FileSet")')
        data = {
            "entity_type": entity_type.lower(),
            "conditions": [c.to_dict() for c in conditions]
        }
        return RestRequest(MetadataApi.query_meta) \
            .with_data(data) \
            .with_credentials() \
            .run()

    @staticmethod
    def get_file_meta(*file_list):
        """Get the metadata for a list of files.

        Directories will be ignored.

        Usage:

        >>> Meta.get_file_meta('/a/b.txt', '/c.json', '/hotpot/eval.py', ...)

        :return:

            .. code-block:: text

                {'data': [{
                            '__create_time__': unix_timestamp,
                            '__creator_id__': int,
                            '__full_path__': str,
                            '__size__': int in bytes,
                            '__type__': str,
                            '__version__': int,
                            '_id': full path with version,
                            'my_meta_key1': 'my_meta_value_1',
                            'tags': ['hotpot', 'cnn', ...]
                           },
                           { ... },
                           ...
                          ],
                 'status': 'success'}
        """

        # Resolve vague paths
        resolved_paths = []
        for f in file_list:
            r = file.File.resolve_vague_path(f)
            if not r['is_dir']:
                resolved_paths.append(r['id'])
        # Get meta data
        data = {
            'type': 'file',
            'ids': resolved_paths
        }
        return RestRequest(MetadataApi.get_meta) \
            .with_data(data) \
            .with_credentials() \
            .run()

    # === DELETE ===
    @staticmethod
    def del_file_meta(file_path, tags: list, keys: list):
        # Resolve the potentially vague path to explicit path
        r = file.File.resolve_vague_path(file_path)

        if r['is_dir']:
            raise AcaiException('Remote path {} is a directory.'
                                'Directories do not have metadata'
                                ''.format(r['id']))
        explicit_path = r['id']
        data = {'id': explicit_path,
                'keys': keys,
                'tags': tags}

        return RestRequest(MetadataApi.del_file_meta) \
            .with_data(data) \
            .with_credentials() \
            .run()

    @staticmethod
    def del_file_set_meta(file_set, tags: list, keys: list):
        # Resolve the potentially vague name to explicit name
        r = fileset.FileSet.resolve_vague_name(file_set)

        explicit_path = r['id']
        data = {'id': explicit_path,
                'keys': keys,
                'tags': tags}

        return RestRequest(MetadataApi.del_file_set_meta) \
            .with_data(data) \
            .with_credentials() \
            .run()

    @staticmethod
    def del_job_meta(job_id: int, tags: list, keys: list):
        data = {'id': job_id,
                'keys': keys,
                'tags': tags}

        return RestRequest(MetadataApi.del_job_meta) \
            .with_data(data) \
            .with_credentials() \
            .run()
