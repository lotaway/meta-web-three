FROM eclipse-temurin:17-jre
WORKDIR /app
COPY target/*.jar message-service.jar
ENTRYPOINT ["java", "-jar", "message-service.jar"]
