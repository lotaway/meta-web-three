# 构建上下文为仓库根目录: docker build -f server/Dockerfile .
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
FROM eclipse-temurin:17-jre-jammy AS product-service
WORKDIR /app
COPY --from=builder /build/server/mall-domain/product-service/target/product-service-*.jar product-service.jar
ENTRYPOINT ["java", "-jar", "product-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS user-service
WORKDIR /app
COPY --from=builder /build/server/mall-domain/user-service/target/user-service-*.jar user-service.jar
ENTRYPOINT ["java", "-jar", "user-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS order-service
WORKDIR /app
COPY --from=builder /build/server/mall-domain/order-service/target/order-service-*.jar order-service.jar
ENTRYPOINT ["java", "-jar", "order-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS cart-service
WORKDIR /app
COPY --from=builder /build/server/mall-domain/cart-service/target/cart-service-*.jar cart-service.jar
ENTRYPOINT ["java", "-jar", "cart-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS payment-service
WORKDIR /app
COPY --from=builder /build/server/mall-domain/payment-service/target/payment-service-*.jar payment-service.jar
ENTRYPOINT ["java", "-jar", "payment-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS promotion-service
WORKDIR /app
COPY --from=builder /build/server/mall-domain/promotion-service/target/promotion-service-*.jar promotion-service.jar
ENTRYPOINT ["java", "-jar", "promotion-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS after-sale-service
WORKDIR /app
COPY --from=builder /build/server/mall-domain/after-sale-service/target/after-sale-service-*.jar after-sale-service.jar
ENTRYPOINT ["java", "-jar", "after-sale-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS group-buying-service
WORKDIR /app
COPY --from=builder /build/server/mall-domain/group-buying-service/target/group-buying-service-*.jar group-buying-service.jar
ENTRYPOINT ["java", "-jar", "group-buying-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS live-service
WORKDIR /app
COPY --from=builder /build/server/mall-domain/live-service/target/live-service-*.jar live-service.jar
ENTRYPOINT ["java", "-jar", "live-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS mall-supplier-service
WORKDIR /app
COPY --from=builder /build/server/mall-domain/mall-supplier-service/target/mall-supplier-service-*.jar mall-supplier-service.jar
ENTRYPOINT ["java", "-jar", "mall-supplier-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS review-service
WORKDIR /app
COPY --from=builder /build/server/mall-domain/review-service/target/review-service-*.jar review-service.jar
ENTRYPOINT ["java", "-jar", "review-service.jar"]

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

FROM eclipse-temurin:17-jre-jammy AS recommendation-service
WORKDIR /app
COPY --from=builder /build/server/mall-domain/recommendation-service/target/recommendation-service-*.jar recommendation-service.jar
ENTRYPOINT ["java", "-jar", "recommendation-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS route-optimizer
WORKDIR /app
COPY --from=builder /build/server/ai-domain/route-optimizer/target/route-optimizer-*.jar route-optimizer.jar
ENTRYPOINT ["java", "-jar", "route-optimizer.jar"]

# supply-chain-domain
FROM eclipse-temurin:17-jre-jammy AS inventory-service
WORKDIR /app
COPY --from=builder /build/server/supply-chain-domain/inventory-service/target/inventory-service-*.jar inventory-service.jar
ENTRYPOINT ["java", "-jar", "inventory-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS warehouse-service
WORKDIR /app
COPY --from=builder /build/server/supply-chain-domain/warehouse-service/target/warehouse-service-*.jar warehouse-service.jar
ENTRYPOINT ["java", "-jar", "warehouse-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS logistics-service
WORKDIR /app
COPY --from=builder /build/server/supply-chain-domain/logistics-service/target/logistics-service-*.jar logistics-service.jar
ENTRYPOINT ["java", "-jar", "logistics-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS procurement-service
WORKDIR /app
COPY --from=builder /build/server/supply-chain-domain/procurement-service/target/procurement-service-*.jar procurement-service.jar
ENTRYPOINT ["java", "-jar", "procurement-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS supplier-service
WORKDIR /app
COPY --from=builder /build/server/supply-chain-domain/supplier-service/target/supplier-service-*.jar supplier-service.jar
ENTRYPOINT ["java", "-jar", "supplier-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS inventory-alert-service
WORKDIR /app
COPY --from=builder /build/server/supply-chain-domain/inventory-alert-service/target/inventory-alert-service-*.jar inventory-alert-service.jar
ENTRYPOINT ["java", "-jar", "inventory-alert-service.jar"]

# erp-domain
FROM eclipse-temurin:17-jre-jammy AS finance-service
WORKDIR /app
COPY --from=builder /build/server/erp-domain/finance-service/target/finance-service-*.jar finance-service.jar
ENTRYPOINT ["java", "-jar", "finance-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS invoice-service
WORKDIR /app
COPY --from=builder /build/server/erp-domain/invoice-service/target/invoice-service-*.jar invoice-service.jar
ENTRYPOINT ["java", "-jar", "invoice-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS reporting-service
WORKDIR /app
COPY --from=builder /build/server/erp-domain/reporting-service/target/reporting-service-*.jar reporting-service.jar
ENTRYPOINT ["java", "-jar", "reporting-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS settlement-service
WORKDIR /app
COPY --from=builder /build/server/erp-domain/settlement-service/target/settlement-service-*.jar settlement-service.jar
ENTRYPOINT ["java", "-jar", "settlement-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS hrm-service
WORKDIR /app
COPY --from=builder /build/server/erp-domain/hrm-service/target/hrm-service-*.jar hrm-service.jar
ENTRYPOINT ["java", "-jar", "hrm-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS project-service
WORKDIR /app
COPY --from=builder /build/server/erp-domain/project-service/target/project-service-*.jar project-service.jar
ENTRYPOINT ["java", "-jar", "project-service.jar"]

# data-pipeline
FROM eclipse-temurin:17-jre-jammy AS data-pipeline
WORKDIR /app
COPY --from=builder /build/server/data-pipeline/target/data-pipeline-*.jar data-pipeline.jar
ENTRYPOINT ["java", "-jar", "data-pipeline.jar"]

# supply-chain-domain (continued)
FROM eclipse-temurin:17-jre-jammy AS rma-service
COPY --from=builder /build/server/supply-chain-domain/rma-service/target/rma-service-*.jar /app/app.jar
WORKDIR /app
EXPOSE 10123
ENTRYPOINT ["java", "-jar", "app.jar"]

FROM eclipse-temurin:17-jre-jammy AS dom-service
COPY --from=builder /build/server/supply-chain-domain/dom-service/target/dom-service-*.jar /app/app.jar
WORKDIR /app
EXPOSE 10124
ENTRYPOINT ["java", "-jar", "app.jar"]

# blockchain-domain
FROM eclipse-temurin:17-jre-jammy AS wallet-service
WORKDIR /app
COPY --from=builder /build/server/blockchain-domain/wallet-service/target/wallet-service-*.jar wallet-service.jar
ENTRYPOINT ["java", "-jar", "wallet-service.jar"]

FROM eclipse-temurin:17-jre-jammy AS traceability-service
WORKDIR /app
COPY --from=builder /build/server/blockchain-domain/traceability-service/target/traceability-service-*.jar traceability-service.jar
ENTRYPOINT ["java", "-jar", "traceability-service.jar"]
