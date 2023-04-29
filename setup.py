import os
import setuptools


def parse_requirements(path):
    """Parse requirements.txt."""
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, path)) as f:
        lines = f.readlines()
    lines = [
        line
        for line in map(lambda l: l.strip(), lines)
        if line != '' and line[0] != '#'
    ]
    return lines


requirements = parse_requirements('requirements.txt')

setuptools.setup(
    name='OoDTool',
    version='0.0',
    packages=setuptools.find_packages(),
    classifiers=[
        # "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        # "Operating System :: OS Independent",
    ],
    install_requires=requirements
)
