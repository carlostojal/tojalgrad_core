FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# update and install dependencies
RUN apt update
RUN apt install build-essential cmake git libeigen3-dev -y

# create directories
RUN mkdir -p /usr/src/tojalgrad_core/lib
RUN mkdir -p /usr/src/tojalgrad_core/cmake-build-debug

# copy code
WORKDIR /usr/src/tojalgrad_core
COPY . .

# clone the gtest src
WORKDIR /usr/src/tojalgrad_core/lib
RUN git clone https://github.com/google/googletest.git

# run cmake
WORKDIR /usr/src/tojalgrad_core/cmake-build-debug
RUN cmake ..
RUN make -j4

# run the tests
ENTRYPOINT ["./test"]

