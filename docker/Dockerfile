# Base Images
## 从天池基础镜像构建
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.4-cuda10.1-py3

## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /

RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-get update
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev
RUN pip install -r requirements.txt

## 镜像启动后统一执行 sh run.sh
# CMD ["sh", "run.sh"]