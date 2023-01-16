from acaisdk.services.api_calls import *

class SparkTuner:
    def __init__(self, jar_path, data_path, cost_per_hour, main_class, job_name, output_path):
        self.jar_path = jar_path
        self.data_path = data_path
        self.cost_per_hour = cost_per_hour
        self.main_class = main_class
        self.job_name = job_name
        self.output_path = output_path

    def construct_params(self, is_create_job: bool):
        params = {}
        if is_create_job:
            params["jar"] = self.jar_path
            params["dataPath"] = self.data_path
            params["costPerHour"] = self.cost_per_hour
            params["mainClass"] = self.main_class
            params["jobName"] = self.job_name
            params["outputPath"] = self.output_path
        else:
            params["jobName"] = self.job_name
        
        return params

    # Eg - curl localhost:8080/accept/job -d jar=gs://cmu-acai-spark-input/pageRank/twitter-etl.jar -d dataPath=gs://cmu-acai-spark-input/twitter_cleaned.parquet -d costPerHour=1 -d mainClass=Twitter -d jobName=twitter -d outputPath=gs://cmu-acai-spark-input/twitterOutput
    def create_job(self):
        r = RestRequest(SparkTunerAcceptApi.job) \
            .with_data(self.construct_params(True)) \
            .with_credentials() \
            .run()

        return r

    # Eg - curl localhost:8080/cluster/tunejob -d jobName=twitter
    def tune_job(self):
        r = RestRequest(SparkTunerTuneApi.tunejob) \
            .with_data(self.construct_params(False)) \
            .with_credentials() \
            .run()

        print('Job has been submitted. Use get_job_status() to get the status')

    # Eg - curl localhost:8080/accept/getstatus -d jobName=twitter
    def get_job_status(self):
        r = RestRequest(SparkTunerAcceptApi.getstatus) \
            .with_data(self.construct_params(False)) \
            .with_credentials() \
            .run()
            
        return r