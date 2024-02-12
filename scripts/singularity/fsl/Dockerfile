FROM ubuntu:18.04
MAINTAINER Ari Kahn arikahn@pennmedicine.upenn.edu

# Core system capabilities required.
# fslreorient2std needs bc
# /usr/local/fsl/bin/standard_space_roi needs dc
# flirt needs libopenblas-dev
RUN apt-get update && apt-get install -y \
dc \
git \
libopenblas-dev \
nodejs \
npm \
python \
tar \
unzip \
wget \
libgl1-mesa-glx \
&& apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN npm install -gq bids-validator

# Run FSL installer and remove unneeded directories
RUN wget -q http://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py && \
chmod 775 fslinstaller.py
RUN /fslinstaller.py -d /usr/local/fsl

# Make FSL happy
ENV FSLDIR=/usr/local/fsl
ENV PATH=$FSLDIR/bin:$PATH
RUN /bin/bash -c 'source /usr/local/fsl/etc/fslconf/fsl.sh'
ENV FSLMULTIFILEQUIT=TRUE
ENV FSLOUTPUTTYPE=NIFTI_GZ
ENV FSLTCLSH=/usr/local/fsl/bin/fsltclsh
ENV FSLWISH=/usr/local/fsl/bin/fslwish

ENV USER=fsluser

# Run configuration script for normal usage
RUN echo ". /usr/local/fsl/etc/fslconf/fsl.sh" >> /root/.bashrc
