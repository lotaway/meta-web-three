FROM eclipse-temurin:17-jre
WORKDIR /app
COPY target/*.jar user-service.jar
ENTRYPOINT ["java", "-jar", "user-service.jar"]
