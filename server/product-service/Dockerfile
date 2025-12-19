FROM eclipse-temurin:17-jre
WORKDIR /app
COPY target/*.jar product-service.jar
ENTRYPOINT ["java", "-jar", "product-service.jar"]
