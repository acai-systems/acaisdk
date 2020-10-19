from acaisdk.services.api_calls import *
from acaisdk import credentials
import json

class AutoML:
    @staticmethod
    def submit_task(filename):
        data = json.dumps(open(filname, 'rb').read())
        r = RestRequest(AutoMLApi.submit_task) \
                .with_data(data) \
                .with_credentials() \
                .run()
        return r
    
    @staticmethod
    def get_status(jobid):
        r = RestRequest(AutoMLApi.job_status) \
            .with_data({'id': [jobid]}) \
            .with_credentials() \
            .run()
        
        return r

