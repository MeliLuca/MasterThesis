FROM nvidia/cuda:10.2-base
CMD nvidia-smi

LABEL author="Luca Melissari"


RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3.6 python3-pip python3-dev
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install git
RUN pip3 -q install pip --upgrade

RUN mkdir src
WORKDIR src/

# Enter root to install
USER root
RUN pip install pandas
RUN pip install numpy
RUN pip install matplotlib
RUN pip install jupyter
RUN pip install opencv-python
RUN pip install scikit-image
RUN pip install tqdm
RUN pip install pytorch-lightning
RUN pip install torch torchvision
USER $NB_UID

#CMD ["jupyter", "notebook", "--port=8887", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
