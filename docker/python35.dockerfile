FROM tensorflow/tensorflow:1.0.0-alpha-py3

RUN apt-get update
RUN apt-get -y install git

RUN pip install pandas
RUN pip install plotly
RUN pip install pdoc
RUN pip install mako
RUN pip install markdown
RUN pip install decorator>=4.0.9
RUN pip install tflearn
RUN pip install asq>=1.2.1
RUN pip install pytest
RUN pip install pytest-sugar

RUN pip install phi>=0.6.4