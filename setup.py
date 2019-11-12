from setuptools import setup, find_packages

setup(name='acaisdk',
      version='0.1',
      description='Acai System SDK',
      url='https://github.com/acai-systems/acaisdk',
      author='Chang Xu',
      author_email='changx@andrew.cmu.edu',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=True)
