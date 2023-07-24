import os
import setuptools
from pathlib import Path


def get_version() -> str:
    """Get the minor version."""
    version_path = Path(
        os.path.realpath(__file__)
    ).parent / "__version__.txt"

    with open(version_path, "r") as version_file:
        version = (version_file.read().split("."))
        return ".".join(version)


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
    name='oodtool',
    version=get_version(),
    packages=setuptools.find_packages(exclude=['oodtool/pyqt_gui/tests', 'oodtool/core/tests']),
    classifiers=[
        # "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        # "Operating System :: OS Independent",
    ],
    package_data={'oodtool': ["oodtool/pyqt_gui/gui_graphics/*.png",
                              "oodtool/core/ood_score/notebooks/OoDExperimental.ipynb"]},
    entry_points={
        'console_scripts': ['oodtool=oodtool.__main__:main']
    },
    install_requires=requirements,
    include_package_data=True,
    extras_require={
        'testing': [
            'parameterized'
        ]
    }
)
