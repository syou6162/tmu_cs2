FROM python:3.7-slim

RUN apt-get -yq update && apt-get install -yq tk-dev git curl jq

WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt -c constraints.txt

RUN make download-kdd-data
RUN python /app/anomaly_detection/trainer.py /app/data/train.pickle

CMD ["/bin/bash"]
