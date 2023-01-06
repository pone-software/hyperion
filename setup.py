from setuptools import setup, find_packages

setup(
    name='hyperion',
    version='0.1.0',
    package_dir={"": "hyperion"},
    packages=find_packages(where='hyperion'),
)