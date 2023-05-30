# tojalgrad core

This is the core library of a simple neural network framework developed in C++.

I plan to add acceleration with OpenCL in a close future.

As of now, it does not contain any autograd functionality, so the name doesn't make sense yet.

DISCLAIMER: I developed this project only to learn some stuff, don't expect stability or many features.

## Requirements

- CMake
- Eigen 3

## Setup

I haven't done the CMake installation rules yet, so the only thing you can do for now is build and run tests :').

### Bare metal

- Create a new directory in this location, for example ```build``` and change to it.
- Run the command ```cmake```.
- Run the command ```make```.
- To run the tests, run ```./test```.

### Docker

In this case, you obviously need Docker installed on your system.

- Run the command ```docker build -t tojalgrad_core_tests .```.
- Run the command ```docker run tojalgrad_core_tests```.