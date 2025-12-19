FROM eclipse-temurin:17-jre
WORKDIR /app
COPY target/*.jar order-service.jar
ENTRYPOINT ["java", "-jar", "order-service.jar"]
