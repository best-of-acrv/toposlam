from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='toposlam',
    version='0.9.0',
    author='Ben Talbot',
    author_email='b.talbot@qut.edu.au',
    url='https://github.com/best-of-acrv/toposlam',
    description=
    'Topographical SLAM using deep visual odomentry & visual place recognition',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=['acrv_datasets'],
    entry_points={'console_scripts': ['toposlam=toposlam.__main__:main']},
    classifiers=(
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ))
