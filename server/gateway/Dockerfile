FROM eclipse-temurin:17-jre
WORKDIR /app
COPY target/*.jar gateway.jar
ENTRYPOINT ["java", "-jar", "gateway.jar"]
