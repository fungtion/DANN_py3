# The dockerfile is built to produce image of the pytorch implementation of Unsupervised Domain Adaptation by Backpropagation

FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

RUN pip install pillow \
 && pip install torchvision \
 && mkdir /DANN

VOLUME ["/DANN/dataset", "/DANN/models"]

EXPOSE 22

COPY ./data_loader.py ./functions.py ./main.py ./model.py ./README.md ./test.py /DANN/

WORKDIR /DANN