FROM iterativeai/cml:0-dvc2-base1-gpu

RUN apt-get update
# Install cv2 dependencies
RUN apt-get install -y ffmpeg libsm6 libxext6

CMD ["cml runner launch", \
     "--repo=$REPOSITORY_URL", \
     "--token=$PERSONAL_ACCESS_TOKEN", \
     "--labels=cml-runner-gpu", \
     "--idle-timeout=never"]
