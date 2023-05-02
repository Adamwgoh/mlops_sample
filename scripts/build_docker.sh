model='$1'
docker build --no-cache -f training/dockerfiles/"${model}".Dockerfile -t "${model}"-"${train}":latest .