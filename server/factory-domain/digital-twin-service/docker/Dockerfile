FROM eclipse-temurin:21-jdk-alpine AS builder

WORKDIR /app

# Step 1: Copy pom.xml and download dependencies (cached layer)
COPY pom.xml .
RUN apk add --no-cache maven && \
    mvn dependency:go-offline -B -DskipTests && \
    apk del maven

# Step 2: Copy source code
COPY src ./src

# Step 3: Build the application
RUN apk add --no-cache maven && \
    mvn clean package -DskipTests && \
    apk del maven

# Final stage
FROM eclipse-temurin:21-jre-alpine

WORKDIR /app

# Create non-root user
RUN addgroup -g 1000 appgroup && \
    adduser -u 1000 -G appgroup -D appuser

# Copy jar file
COPY --from=builder /app/target/*.jar app.jar

# Set ownership
RUN chown -R appuser:appgroup /app

USER appuser

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/actuator/health || exit 1

# JVM options via environment variable (PID 1 is Java)
ENV JAVA_TOOL_OPTIONS="-Xms512m -Xmx1024m -XX:+UseG1GC -XX:MaxGCPauseMillis=200"

# Use exec form - Java receives signals directly
ENTRYPOINT ["java", "-jar", "app.jar"]