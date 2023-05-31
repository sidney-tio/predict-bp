FROM public.ecr.aws/lambda/python:3.9

COPY . ${LAMBDA_TASK_ROOT}
COPY requirements.txt .

RUN yum install mesa-libGL -y
RUN yum install libsndfile -y
RUN pip3 install --upgrade pip &&\
    pip3 install -r requirements.txt

CMD [ "handler.handler" ]
