FROM ubuntu:22.04

WORKDIR /project

COPY requirements.txt .

RUN apt-get update && apt-get install -y python3 python3-pip

RUN pip install --no-cache-dir -r requirements.txt

ADD experiment_files/ experiment_files/

RUN mkdir experiment_res/

CMD nvidia-smi;export TERM=xterm && top -b | head; ./experiment_files/run_sript.sh
