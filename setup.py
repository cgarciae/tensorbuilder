import os
from setuptools import setup
from pip.req import parse_requirements

# parse requirements
reqs = [str(r.req) for r in parse_requirements("requirements.txt", session=False)]


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

version = read('tensorbuilder/version.txt').split("\n")[0]

setup(
    name = "tensorbuilder",
    version = version,
    author = "Cristian Garcia",
    author_email = "cgarcia.e88@gmail.com",
    description = ("A light wrapper over TensorFlow that enables you to easily create complex deep neural networks using the Builder Pattern through a functional fluent immutable API"),
    license = "MIT",
    keywords = ["tensorflow", "deep learning", "neural networks"],
    url = "https://github.com/cgarciae/tensorbuilder",
   	packages = [
        'tensorbuilder',
        'tensorbuilder.tensordata',
        'tensorbuilder.patches',
        'tensorbuilder.tests'
    ],
    package_data={
        '': ['LICENCE', 'requirements.txt', 'README.md', 'CHANGELOG.md'],
        'tensorbuilder': ['version.txt', 'README-template.md']
    },
    download_url = 'https://github.com/cgarciae/tensorbuilder/tarball/{0}'.format(version),
    include_package_data = True,
    long_description = read('README.md'),
    install_requires = reqs
)
