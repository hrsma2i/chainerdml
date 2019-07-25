from setuptools import setup, find_packages


with open('requirements.txt') as requirements_file:
    install_requires = requirements_file.read().splitlines()

setup(
    author='hrsma2i',
    author_email='hrs.ma2i@gmail.com',
    name='chainerdml',
    description='a Chainer-based library for Deep Metric Learning',
    version='0.1.0a1',
    packages=find_packages(),
    install_requires=install_requires,
    license='MIT license',
)
