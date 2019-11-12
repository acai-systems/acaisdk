import os
from setuptools import setup, find_packages




def parse_requirements():
    requirements = []
    r_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'acaisdk/requirements.txt')
    with open(r_file) as f:
        for l in f:
            if l.strip():
                requirements.append(l.strip())

    requirements.append('acaisdk')
    return requirements


setup(name='acaisdk',
      version='0.1',
      description='Acai System SDK',
      url='https://github.com/acai-systems/acaisdk',
      author='Chang Xu',
      author_email='changx@andrew.cmu.edu',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      install_requires=parse_requirements(),
      zip_safe=True)
