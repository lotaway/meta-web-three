FROM openjdk:21-jdk-slim
WORKDIR /app
COPY target/project-service-1.0.0.jar app.jar
EXPOSE 10089
ENTRYPOINT ["java", "-jar", "app.jar"]