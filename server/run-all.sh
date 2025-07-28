# run-all.sh
docker run -d -name zookeeper -p 2181:2181 --restart unless-stopped zookeeper:3.7 &
cd product-service && mvn spring-boot:run &
cd user-service && mvn spring-boot:run &
cd message-service && mvn spring-boot:run &
cd order-service && mvn spring-boot:run &