#Deriving the latest base image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

#Labels as key value pair
LABEL Maintainer="nico.reeb"

# export timezone - for python3.9-dev install
ENV TZ=Europe/Berlin \
  DEBIAN_FRONTEND=noninteractive

# place timezone data /etc/timezone
#RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y python3 python3-pip python3-tk libx11-dev tk

#
# PyCUDA
#
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
RUN echo "$PATH" && echo "$LD_LIBRARY_PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir numpy matplotlib pycuda scipy cupy pandas

#RUN addgroup --gid 1006 user
#RUN adduser --disabled-password --gecos '' --uid 1000 --gid 1006 user
RUN adduser user
USER user
RUN mkdir /home/user/code
COPY . /home/user/code
RUN pip install -e /home/user/code/.
