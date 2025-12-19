FROM eclipse-temurin:17-jre
WORKDIR /app
COPY target/*.jar media-service.jar
ENTRYPOINT ["java", "-jar", "media-service.jar"]
