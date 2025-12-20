FROM maven:3.9.11-eclipse-temurin-17 AS builder
WORKDIR /build
COPY pom.xml .
RUN mvn -B install -DskipTests
COPY */pom.xml ./
RUN mvn -B -e -C dependency:go-offline
COPY . .

RUN mvn -B package -DskipTests

FROM eclipse-temurin:17-jre-jammy AS gateway
WORKDIR /app
COPY --from=builder /build/gateway/target/gateway-*.jar gateway.jar
ENTRYPOINT ["java", "-jar", "gateway.jar"]

FROM eclipse-temurin:17-jre-jammy AS product-service
WORKDIR /app
COPY --from=builder /build/product-service/target/product-service-*.jar product-service.jar
ENTRYPOINT ["java", "-jar", "product-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS user-service
WORKDIR /app
COPY --from=builder /build/user-service/target/user-service-*.jar user-service.jar
ENTRYPOINT ["java", "-jar", "user-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS order-service
WORKDIR /app
COPY --from=builder /build/order-service/target/order-service-*.jar order-service.jar
ENTRYPOINT ["java", "-jar", "order-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS message-service
WORKDIR /app
COPY --from=builder /build/message-service/target/message-service-*.jar message-service.jar
ENTRYPOINT ["java", "-jar", "message-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS payment-service
WORKDIR /app
COPY --from=builder /build/payment-service/target/payment-service-*.jar payment-service.jar
ENTRYPOINT ["java", "-jar", "payment-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS media-service
WORKDIR /app
COPY --from=builder /build/media-service/target/media-service-*.jar media-service.jar
ENTRYPOINT ["java", "-jar", "media-service.jar"]
