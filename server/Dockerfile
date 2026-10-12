# 构建上下文为仓库根目�? docker build -f server/Dockerfile .
FROM maven:3.9.11-eclipse-temurin-17 AS builder
WORKDIR /build
COPY shared/event-sdk ./shared/event-sdk
COPY server ./server
RUN mvn -B -f shared/event-sdk/pom.xml install -DskipTests
RUN mvn -B -f server/pom.xml install -DskipTests

FROM eclipse-temurin:17-jre-jammy AS gateway
WORKDIR /app
COPY --from=builder /build/server/gateway/target/gateway-*.jar gateway.jar
ENTRYPOINT ["java", "-jar", "gateway.jar"]

# mall-domain
FROM eclipse-temurin:17-jre-jammy AS mall-catalog-service
WORKDIR /app
COPY --from=builder /build/server/mall-domain/mall-catalog-service/target/mall-catalog-service-*.jar mall-catalog-service.jar
ENTRYPOINT ["java", "-jar", "mall-catalog-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS mall-user-service
WORKDIR /app
COPY --from=builder /build/server/mall-domain/mall-user-service/target/mall-user-service-*.jar mall-user-service.jar
ENTRYPOINT ["java", "-jar", "mall-user-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS mall-trade-service
WORKDIR /app
COPY --from=builder /build/server/mall-domain/mall-trade-service/target/mall-trade-service-*.jar mall-trade-service.jar
ENTRYPOINT ["java", "-jar", "mall-trade-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS mall-social-service
WORKDIR /app
COPY --from=builder /build/server/mall-domain/mall-social-service/target/mall-social-service-*.jar mall-social-service.jar
ENTRYPOINT ["java", "-jar", "mall-social-service.jar"]

# platform-domain
FROM eclipse-temurin:17-jre-jammy AS platform-service
WORKDIR /app
COPY --from=builder /build/server/platform-domain/platform-service/target/platform-service-*.jar platform-service.jar
ENTRYPOINT ["java", "-jar", "platform-service.jar"]

# factory-domain
FROM eclipse-temurin:17-jre-jammy AS factory-service
WORKDIR /app
COPY --from=builder /build/server/factory-domain/factory-service/target/factory-service-*.jar factory-service.jar
ENTRYPOINT ["java", "-jar", "factory-service.jar"]

# ai-domain
FROM eclipse-temurin:17-jre-jammy AS ai-service
WORKDIR /app
COPY --from=builder /build/server/ai-domain/ai-service/target/ai-service-*.jar ai-service.jar
ENTRYPOINT ["java", "-jar", "ai-service.jar"]

# supply-chain-domain
FROM eclipse-temurin:17-jre-jammy AS scm-inventory-service
WORKDIR /app
COPY --from=builder /build/server/supply-chain-domain/scm-inventory-service/target/scm-inventory-service-*.jar scm-inventory-service.jar
ENTRYPOINT ["java", "-jar", "scm-inventory-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS scm-logistics-service
WORKDIR /app
COPY --from=builder /build/server/supply-chain-domain/scm-logistics-service/target/scm-logistics-service-*.jar scm-logistics-service.jar
ENTRYPOINT ["java", "-jar", "scm-logistics-service.jar"]

# erp-domain
FROM eclipse-temurin:17-jre-jammy AS finance-service
WORKDIR /app
COPY --from=builder /build/server/erp-domain/finance-service/target/finance-service-*.jar finance-service.jar
ENTRYPOINT ["java", "-jar", "finance-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS erp-management-service
WORKDIR /app
COPY --from=builder /build/server/erp-domain/erp-management-service/target/erp-management-service-*.jar erp-management-service.jar
ENTRYPOINT ["java", "-jar", "erp-management-service.jar"]



# data-pipeline
FROM eclipse-temurin:17-jre-jammy AS data-pipeline
WORKDIR /app
COPY --from=builder /build/server/data-pipeline/target/data-pipeline-*.jar data-pipeline.jar
ENTRYPOINT ["java", "-jar", "data-pipeline.jar"]

# blockchain-domain
FROM eclipse-temurin:17-jre-jammy AS blockchain-service
WORKDIR /app
COPY --from=builder /build/server/blockchain-domain/blockchain-service/target/blockchain-service-*.jar blockchain-service.jar
ENTRYPOINT ["java", "-jar", "blockchain-service.jar"]
