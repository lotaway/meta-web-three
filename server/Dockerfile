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

FROM eclipse-temurin:17-jre-jammy AS user-service
WORKDIR /app
COPY --from=builder /build/server/mall-domain/user-service/target/user-service-*.jar user-service.jar
ENTRYPOINT ["java", "-jar", "user-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS mall-trade-service
WORKDIR /app
COPY --from=builder /build/server/mall-domain/mall-trade-service/target/mall-trade-service-*.jar mall-trade-service.jar
ENTRYPOINT ["java", "-jar", "mall-trade-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS group-buying-service
WORKDIR /app
COPY --from=builder /build/server/mall-domain/group-buying-service/target/group-buying-service-*.jar group-buying-service.jar
ENTRYPOINT ["java", "-jar", "group-buying-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS live-service
WORKDIR /app
COPY --from=builder /build/server/mall-domain/live-service/target/live-service-*.jar live-service.jar
ENTRYPOINT ["java", "-jar", "live-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS risk-control-service
WORKDIR /app
COPY --from=builder /build/server/mall-domain/risk-control-service/target/risk-control-service-*.jar risk-control-service.jar
ENTRYPOINT ["java", "-jar", "risk-control-service.jar"]

# platform-domain
FROM eclipse-temurin:17-jre-jammy AS message-service
WORKDIR /app
COPY --from=builder /build/server/platform-domain/message-service/target/message-service-*.jar message-service.jar
ENTRYPOINT ["java", "-jar", "message-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS commission-service
WORKDIR /app
COPY --from=builder /build/server/platform-domain/commission-service/target/commission-service-*.jar commission-service.jar
ENTRYPOINT ["java", "-jar", "commission-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS media-service
WORKDIR /app
COPY --from=builder /build/server/platform-domain/media-service/target/media-service-*.jar media-service.jar
ENTRYPOINT ["java", "-jar", "media-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS user-action-service
WORKDIR /app
COPY --from=builder /build/server/platform-domain/user-action-service/target/user-action-service-*.jar user-action-service.jar
ENTRYPOINT ["java", "-jar", "user-action-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS cs-service
WORKDIR /app
COPY --from=builder /build/server/platform-domain/cs-service/target/cs-service-*.jar cs-service.jar
ENTRYPOINT ["java", "-jar", "cs-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS data-analysis-service
WORKDIR /app
COPY --from=builder /build/server/platform-domain/data-analysis-service/target/data-analysis-service-*.jar data-analysis-service.jar
ENTRYPOINT ["java", "-jar", "data-analysis-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS social-commerce-service
WORKDIR /app
COPY --from=builder /build/server/platform-domain/social-commerce-service/target/social-commerce-service-*.jar social-commerce-service.jar
ENTRYPOINT ["java", "-jar", "social-commerce-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS developer-portal-service
WORKDIR /app
COPY --from=builder /build/server/platform-domain/developer-portal-service/target/developer-portal-service-*.jar developer-portal-service.jar
ENTRYPOINT ["java", "-jar", "developer-portal-service.jar"]

# factory-domain
FROM eclipse-temurin:17-jre-jammy AS mes-service
WORKDIR /app
COPY --from=builder /build/server/factory-domain/mes-service/target/mes-service-*.jar mes-service.jar
ENTRYPOINT ["java", "-jar", "mes-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS digital-twin-service
WORKDIR /app
COPY --from=builder /build/server/factory-domain/digital-twin-service/target/digital-twin-service-*.jar digital-twin-service.jar
ENTRYPOINT ["java", "-jar", "digital-twin-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS production-service
WORKDIR /app
COPY --from=builder /build/server/factory-domain/production-service/target/production-service-*.jar production-service.jar
ENTRYPOINT ["java", "-jar", "production-service.jar"]

# ai-domain
FROM eclipse-temurin:17-jre-jammy AS forecasting-service
WORKDIR /app
COPY --from=builder /build/server/ai-domain/forecasting-service/target/forecasting-service-*.jar forecasting-service.jar
ENTRYPOINT ["java", "-jar", "forecasting-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS route-optimizer
WORKDIR /app
COPY --from=builder /build/server/ai-domain/route-optimizer/target/route-optimizer-*.jar route-optimizer.jar
ENTRYPOINT ["java", "-jar", "route-optimizer.jar"]

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

# platform-domain (continued)
FROM eclipse-temurin:17-jre-jammy AS tenant-service
WORKDIR /app
COPY --from=builder /build/server/platform-domain/tenant-service/target/tenant-service-*.jar tenant-service.jar
ENTRYPOINT ["java", "-jar", "tenant-service.jar"]

# data-pipeline
FROM eclipse-temurin:17-jre-jammy AS data-pipeline
WORKDIR /app
COPY --from=builder /build/server/data-pipeline/target/data-pipeline-*.jar data-pipeline.jar
ENTRYPOINT ["java", "-jar", "data-pipeline.jar"]

# blockchain-domain
FROM eclipse-temurin:17-jre-jammy AS wallet-service
WORKDIR /app
COPY --from=builder /build/server/blockchain-domain/wallet-service/target/wallet-service-*.jar wallet-service.jar
ENTRYPOINT ["java", "-jar", "wallet-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS traceability-service
WORKDIR /app
COPY --from=builder /build/server/blockchain-domain/traceability-service/target/traceability-service-*.jar traceability-service.jar
ENTRYPOINT ["java", "-jar", "traceability-service.jar"]
