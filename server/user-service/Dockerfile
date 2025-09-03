FROM openjdk:17
WORKDIR /app
COPY target/*.jar user-service.jar
ENTRYPOINT ["java", "-jar", "user-service.jar"]
