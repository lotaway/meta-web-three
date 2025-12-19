FROM openjdk:17
WORKDIR /app
COPY target/*.jar media-service.jar
ENTRYPOINT ["java", "-jar", "media-service.jar"]
