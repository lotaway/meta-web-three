FROM openjdk:17
WORKDIR /app
COPY target/*.jar order-service.jar
ENTRYPOINT ["java", "-jar", "order-service.jar"]
