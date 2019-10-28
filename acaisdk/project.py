from acaisdk.services.api_calls import *


class Project:
    @staticmethod
    def create_project(project_id: str,
                       admin_token: str,
                       project_admin: str) -> dict:
        """This is the starting point of your ACAI journey.

        Project, like its definition in GCP, is a bundle of resources. Users,
        files and jobs are only identifiable when ACAI system knows which
        project they are under.

        Use this method to create a project.

        :param project_id:
            Name of the project, it should be unique, as it is also the ID
            ACAI uses to identify a project.
        :param admin_token:
            One token to rule them all. This is the admin token to create
            new projects.
        :param project_admin:
            An user name for the project administrator.
        :return:

            .. code-block::

                {
                  "admin_token": "string",
                  "project_id": "string",
                  "project_admin_name": "string"
                }
        """
        return RestRequest(CredentialApi.create_project) \
            .with_data({'project_id': project_id,
                        'admin_token': admin_token,
                        "project_admin_name": project_admin}) \
            .run()

    @staticmethod
    def create_user(project_id: str,
                    admin_token: str,
                    user: str) -> dict:
        """Create a new user for the project.

        :param project_id:
            Project ID.
        :param admin_token:
            Use the admin token you get from :py:meth:`~Project.create_project`
        :param user:
            Name for the new user.
        :return:

            .. code-block::

                {
                  "user_id": 0,
                  "user_token": "string"
                }
        """
        # Admin could be global or project admin
        return RestRequest(CredentialApi.create_user) \
            .with_data({'project_id': project_id,
                        'admin_token': admin_token,
                        "user_name": user}) \
            .run()
