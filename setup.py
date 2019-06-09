import sys, os

from setuptools import setup
from setuptools.command.test import test as TestCommand


# Needed packages from requirements.txt
with open('./requirements.txt', mode='r') as file:
    packages = file.readlines().split('\n')


setup(
    name='detection template',
    version='0.0.1',
    description='a template of object detection for ABEJA Platform',
    author='Yusuke Kominami',
    author_email='yukke.konan@gmail.com',
    install_requires=packages,
    url='https://github.com/abeja-inc/platform-template-object-detection/',
    license='MIT License',
    pakcage=find_packages(exclude=('tests', 'docs', '.abeja', '.github'))
)