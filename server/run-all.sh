# run-all.sh
cd product-service && mvn spring-boot:run &
cd user-service && mvn spring-boot:run &
cd message-service && mvn spring-boot:run &
cd order-service && mvn spring-boot:run &