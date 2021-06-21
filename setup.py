from setuptools import find_packages, setup

setup(
    name="TFRecord_IO",
    version="0.1",
    description="Utilities to read and write tfrecord files",
    url="https://github.com/adriankoering/tfrecord_io",
    author="Adrian Köring",
    license="MIT",
    packages=find_packages(),
    install_requires=["pillow", "pathlib"],
)
