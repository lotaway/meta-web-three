cd server && mvn clean install &
cd server/common && mvn clean install &
cd server/product-service && mvn spring-boot:run &
cd server/user-service && mvn spring-boot:run &
cd server/message-service && mvn spring-boot:run &
cd server/order-service && mvn spring-boot:run &
cd server/payment-service && mvn spring-boot:run &
cd server/commission-service && mvn spring-boot:run &
cd server/media-service && mvn spring-boot:run
