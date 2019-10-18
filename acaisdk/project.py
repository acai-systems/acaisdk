from acaisdk.services.api_calls import *


class Project:
    @staticmethod
    def create_project(project_id: str,
                       admin_token: str,
                       project_admin: str):
        return RestRequest(Credential.create_project) \
            .with_data({'project_id': project_id,
                        'admin_token': admin_token,
                        "project_admin_name": project_admin}) \
            .run()

    @staticmethod
    def create_user(project_id: str,
                    admin_token: str,
                    user: str):
        # Admin could be global or project admin
        return RestRequest(Credential.create_user) \
            .with_data({'project_id': project_id,
                        'admin_token': admin_token,
                        "user_name": user}) \
            .run()
