FROM nvcr.nju.edu.cn/nvidia/pytorch:24.10-py3

RUN pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
RUN apt update && apt install git -y

RUN git clone https://github.com/ModelTC/MQBench.git && \
    cd MQBench && \
    pip install -e .

WORKDIR /workspaces

RUN pip install easydict
RUN pip install opencv-python==4.8.0.74 lvis scikit-image tensorboard tensorboardX