FROM nvidia/cuda:12.0.1-devel-ubuntu22.04
RUN ln -s /usr/bin/python3.8 /usr/local/bin/python
ARG USERNAME=root
ARG UID
ARG GROUPNAME
ARG GID
ARG USE_SHELL=/bin/bash
ARG ROOT_PASSWORD=admin

# python setup
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get -y install nano vim wget curl unzip openssh-server sudo git make
RUN apt-get update && apt-get install -y --no-install-recommends wget build-essential libreadline-dev \
libncursesw5-dev libssl-dev libsqlite3-dev libgdbm-dev libbz2-dev liblzma-dev zlib1g-dev uuid-dev libffi-dev libdb-dev
RUN wget --no-check-certificate https://www.python.org/ftp/python/3.8.16/Python-3.8.16.tgz \
&& tar -xzf Python-3.8.16.tgz \
&& cd Python-3.8.16 \
&& ./configure --enable-optimizations\
&& make \
&& sudo make install


# c++
RUN apt-get -y install software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get -y install libgmp-dev libmpfr-dev libmpc-dev libppl-dev
# libcloog-ppl-dev
RUN apt-get update
RUN apt-get install -y gcc-9
RUN apt-get install -y g++-9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 30
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 30

# python libs
RUN pip3 install --upgrade pip
# openexr
RUN apt-get -y install libopenexr-dev
RUN apt-get -y install openexr
RUN pip3 install black==21.10b0 flake8==4.0.1 isort==5.10.1 mypy==0.910 click==8.0.4

COPY ./requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install pickle5


RUN wget http://raw.githubusercontent.com/edihbrandon/RictyDiminished/master/RictyDiminished-Regular.ttf -P /home
RUN sudo apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6
RUN pip3 install git+https://github.com/jamesbowman/openexrpython.git