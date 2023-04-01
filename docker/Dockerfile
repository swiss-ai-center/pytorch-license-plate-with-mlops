FROM iterativeai/cml:0-dvc2-base1-gpu

# Install git?

CMD ["cml runner launch", \
     "--repo=$REPOSITORY_URL", \
     "--token=$PERSONAL_ACCESS_TOKEN", \
     "--labels=cml-gpu", \
     "--idle-timeout=never"]
