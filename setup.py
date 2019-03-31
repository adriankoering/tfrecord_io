from setuptools import find_packages, setup

setup_args = generate_distutils_setup(
    name="TFRecord_IO",
    version="0.1",
    packages=find_packages(),
    install_requires=["tensorflow", "PIL", "pathlib"]
)
setup(**setup_args)
