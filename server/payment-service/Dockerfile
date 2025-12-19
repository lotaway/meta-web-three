FROM openjdk:17

WORKDIR /app

# Copy the JAR file
COPY target/payment-service-*.jar payment-service.jar

# Expose the port
EXPOSE 10086

# Set environment variables
ENV JAVA_OPTS="-Xmx512m -Xms256m"

# Run the application
ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar payment-service.jar"]