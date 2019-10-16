from services.api_calls import *


class Project:
    @staticmethod
    def create_project(project_id: str, admin_token: str):
        """
        :return:    {
                      "project_id": "string",
                      "user_token": "string"
                    }
        """
        r = RestRequest(Credential.create_project) \
            .with_data({'project_id': project_id,
                        'admin_token': admin_token,
                        "project_admin_name": "Chang"}) \
            .run()
        return r

    # @staticmethod
    # def create_user():
