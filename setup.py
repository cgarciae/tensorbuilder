from setuptools import setup, find_packages

setup(
    name='tensorbuilder',
    version='1.0.0',
    description="""TensorBuilder is light wrapper over TensorFlow that enables you to easily create complex deep neural networks 
                   using the Builder Pattern through a functional fluent immutable API.""",
    url='https://github.com/cgarciae/tensorbuilder',
    license='MIT',
    packages=find_packages(include=['tensorbuilder']),
    install_requires=['decorator'],
)
