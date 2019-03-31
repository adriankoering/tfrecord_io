from setuptools import find_packages, setup

setup(
    name="TFRecord_IO",
    version="0.1",
    description='The funniest joke in the world',
    url='https://github.com/adriankoering/tfrecord_io',
    author='Adrian Köring',
    license='MIT',
    packages=find_packages(),
    install_requires=["tensorflow", "pillow", "pathlib"]
)
