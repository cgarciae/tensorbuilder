FROM tensorflow/tensorflow:0.10.0rc0

RUN apt-get update
RUN apt-get -y install git

RUN pip install prettytensor
RUN pip install pandas
RUN pip install plotly
RUN pip install pdoc
RUN pip install mako
RUN pip install markdown
RUN pip install decorator==4.0.9
RUN pip install git+https://github.com/tflearn/tflearn.git
RUN pip install asq==1.2.1
RUN pip install pytest
RUN pip install pytest-sugar
