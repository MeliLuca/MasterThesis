FROM nvidia/cuda:10.2-base
CMD nvidia-smi

LABEL author="Luca Melissari"


RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3.6 python3-pip python3-dev
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y git
RUN pip3 -q install pip --upgrade

RUN mkdir src
WORKDIR src/

# Enter root to install
USER root
RUN pip3 install torch torchvision
RUN pip3 install pytorch-lightning
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install matplotlib
RUN pip3 install jupyter
RUN pip3 install opencv-python
RUN pip3 install scikit-learn
RUN pip3 install scikit-image
RUN pip3 install tqdm
RUN pip3 install gpustat
USER $NB_UID

#CMD ["jupyter", "notebook", "--port=8887", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
