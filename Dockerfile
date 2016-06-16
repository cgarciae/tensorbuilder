FROM tensorflow/tensorflow:r0.9rc0

RUN apt-get update
RUN apt-get install -y \
  git

RUN pip install prettytensor
RUN pip install pandas
RUN pip install plotly
RUN pip install pdoc
RUN pip install mako
RUN pip install markdown
RUN pip install decorator==4.0.9
RUN pip install git+https://github.com/tflearn/tflearn.git
RUN pip install asq==1.2.1
