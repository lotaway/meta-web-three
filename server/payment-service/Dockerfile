FROM maven:3.9.11-openjdk-17 AS builder
WORKDIR /app
COPY pom.xml .
COPY src ./src
RUN mvn clean package -DskipTests

FROM eclipse-temurin:17-jre
WORKDIR /app
COPY --from=builder /app/target/payment-service-*.jar payment-service.jar
ENTRYPOINT ["java", "-jar", "payment-service.jar"]
