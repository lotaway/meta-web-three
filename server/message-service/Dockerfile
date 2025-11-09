FROM openjdk:17
WORKDIR /app
COPY target/*.jar message-service.jar
ENTRYPOINT ["java", "-jar", "message-service.jar"]
