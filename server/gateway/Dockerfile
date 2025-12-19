FROM openjdk:17
WORKDIR /app
COPY target/*.jar gateway.jar
ENTRYPOINT ["java", "-jar", "gateway.jar"]
