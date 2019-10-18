from acaisdk.services.api_calls import *
from typing import Iterable, List
from acaisdk import fileset


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
        return RestRequest(Metadata.query_meta) \
            .with_data(data) \
            .with_credentials() \
            .run()

    @staticmethod
    def get_file_meta(file_list: List[str]):
        """file_list can be of vague name"""

        # Resolve vague paths
        resolved_paths = []
        for f in file_list:
            r = fileset.FileSet.resolve_vague_path(f)
            if not r['is_dir']:
                resolved_paths.append(r["version_specific_path"])

        # Get meta data
        data = {
            'type': 'file',
            'ids': resolved_paths
        }
        return RestRequest(Metadata.get_meta) \
            .with_data(data) \
            .with_credentials() \
            .run()
