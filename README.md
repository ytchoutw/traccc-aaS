# Traccc as-a-Service

Main objective of this repo: run [traccc](https://github.com/acts-project/traccc/tree/main) as-a-Service. Getting this working includes creating three main components:

1. a shared library of `traccc` and writing a standalone version with the essential pieces of the code included
2. a custom backend using the standalone version above to launch the trition server
3. a client to send data to the server

A minimal description of how to build a working version is detailed below. In each subdirectory of this project, a README containing more information can be found. 

## Previous work

The beginnings of this work is based on the CPU version included here developed by Haoran Zhao. The original repo can be found [here](https://github.com/hrzhao76/traccc-aaS). This CPU version has been incorporated into the workflow here such that both a CPU and GPU version are available. 

## Running out of the box

### Get the code

Simply clone the repository with 

```
git clone --recurse-submodules git@github.com:milescb/traccc-aaS.git
```

_Note: an old release of `traccc` is used to match performance plots. In order to get the proper version for running do_

```
cd traccc && git checkout v0.10.0
```

### Docker

A docker built for the triton server can be found at `docexoty/tritonserver:latest`. To run this do

```
shifter --module=gpu --image=docexoty/tritonserver:latest
```

or use your favorite docker application and mount the appropriate directories. 

### Shared Library 

To run out of the box, an installation of `traccc` and the the backend can be found at `/global/cfs/projectdirs/m3443/data/traccc-aaS/software/prod/ver_08282024_traccc_v0.10.0/install`. To set up the environment, run the docker then set the following environment variables

```
export DATADIR=/global/cfs/projectdirs/m3443/data/traccc-aaS/data
export INSTALLDIR=/global/cfs/projectdirs/m3443/data/traccc-aaS/software/prod/ver_08282024_traccc_v0.10.0/install
export PATH=$INSTALLDIR/bin:$PATH
export LD_LIBRARY_PATH=$INSTALLDIR/lib:$LD_LIBRARY_PATH
```

Then the server can be launched with 

```
tritonserver --model-repository=$INSTALLDIR/models
```

Once the server is launched, run the model via:

```
cd client && python TracccTritionClient.py 
```
The `--architecture` tag can be used to toggle the cpu and gpu version via `-a cpu`, for instance. More info in the client directory. 

## Building the backend

First, enter the docker and set environment variables as documented above. Then run

```
cd backend/traccc-gpu && mkdir build install && cd build
cmake -B . -S ../ \
    -DCMAKE_INSTALL_PREFIX=../install/ \
    -DCMAKE_INSTALL_PREFIX=../install/

cmake --build . --target install -- -j20
```

Both the CPU and GPU versions must be built separately. Then, the server can be launched as above:

```
tritonserver --model-repository=../../models
```