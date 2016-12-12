#! /usr/bin/python

import os

print("Getting Version")
with open('tensorbuilder/version.txt', 'r') as f:
    version = f.read()

print("Getting Readme Template")
with open('tensorbuilder/README-template.md', 'r') as f:
    readme = f.read().format(version)

print("Writting Readme")
with open('README.md', 'w') as f:
    f.write(readme)

