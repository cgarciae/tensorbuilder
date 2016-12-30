import os


with open('tensorbuilder/version.txt', 'r') as f:
    version = f.read()

with open('tensorbuilder/README-template.md', 'r') as f:
    readme = f.read().format(version)


with open('README.md', 'w') as f:
    f.write(readme)

