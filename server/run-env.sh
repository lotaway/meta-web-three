# run-env.sh
docker run -d -name zookeeper -p 2181:2181 --restart unless-stopped zookeeper:3.7