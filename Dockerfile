FROM gcr.io/tensorflow/tensorflow

RUN pip install prettytensor
RUN pip install pandas
RUN pip install plotly
RUN pip install pdoc
RUN pip install mako
RUN pip install markdown
RUN pip install decorator==4.0.9