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
    def __init__(self, key: str):
        self.key = key
        self.value = None
        self.type = None

    def value(self, value):
        self.type = ConditionType.VALUE
        self.value = value
        return self

    def max(self):
        self.type = ConditionType.MAX
        return self

    def min(self):
        self.type = ConditionType.MIN
        return self

    def range(self, start, end):
        self.type = ConditionType.RANGE
        self.value = [start, end]
        return self

    def array(self, array: Iterable):
        self.type = ConditionType.ARRAY
        self.value = list(array)
        return self

    def to_dict(self):
        d = {
            "key": self.key,
            "type": str(self.type)
        }
        if self.value:
            d['value'] = self.value
        return d


class Meta:
    # === UPDATE ===
    @staticmethod
    def update_file_meta(file_path,
                         tags: list = None,
                         kv_pairs: dict = None,
                         **kwargs):
        # Resolve the potentially vague path to explicit path
        r = file.File.resolve_vague_path(file_path)

        if r['is_dir']:
            raise AcaiException('Remote path {} is a directory.'
                                'Directories do not have metadata'
                                ''.format(r['id']))
        explicit_path = r['id']
        all_kv_pairs = Meta._merge_dict(kv_pairs, kwargs)
        data = {'id': explicit_path,
                'meta': all_kv_pairs}
        if tags:
            data['tags'] = tags

        return RestRequest(MetadataApi.update_file_meta) \
            .with_data(data) \
            .with_credentials() \
            .run()

    @staticmethod
    def update_file_set_meta(file_set,
                             tags: list = None,
                             kv_pairs: dict = None,
                             **kwargs):
        # Resolve the potentially vague name to explicit name
        r = fileset.FileSet.resolve_vague_name(file_set)

        explicit_name = r['id']
        all_kv_pairs = Meta._merge_dict(kv_pairs, kwargs)
        data = {'id': explicit_name,
                'meta': all_kv_pairs}
        if tags:
            data['tags'] = tags

        return RestRequest(MetadataApi.update_file_set_meta) \
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

    # === DELETE ===
    @staticmethod
    def del_file_meta(file_path, keys: list, tags: list):
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
    def del_file_set_meta(file_set, keys: list, tags: list):
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

    # === SEARCHING ===
    @staticmethod
    def find_file(*conditions: Condition):
        return Meta.query_meta('file', *conditions)

    @staticmethod
    def find_job(*conditions: Condition):
        return Meta.query_meta('job', *conditions)

    @staticmethod
    def find_file_set(*conditions: Condition):
        return Meta.query_meta('fileset', *conditions)

    @staticmethod
    def query_meta(entity_type: str, *conditions: Condition):
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
    def get_file_meta(file_list: List[str]):
        """file_list can be of vague name"""

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
