package com.metawebthree.mes.application.event;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.mes.domain.service.MesDomainService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

import java.util.Map;

/**
 * ERP production order event listener.
 * Listens for production order release events from ERP/production-service
 * and automatically creates corresponding MES work orders.
 */
@Component
public class ProductionOrderEventListener {

    private static final Logger log = LoggerFactory.getLogger(ProductionOrderEventListener.class);

    private final MesDomainService mesDomainService;
    private final ObjectMapper objectMapper;

    public ProductionOrderEventListener(MesDomainService mesDomainService,
                                        ObjectMapper objectMapper) {
        this.mesDomainService = mesDomainService;
        this.objectMapper = objectMapper;
    }

    @KafkaListener(topics = "production.events", groupId = "mes-service")
    public void onProductionEvent(String message) {
        try {
            log.info("Received production event: {}", message);
            Map<String, Object> eventData = objectMapper.readValue(message, Map.class);

            String event = (String) eventData.get("event");
            if (event == null) {
                log.warn("Unknown production event format: {}", message);
                return;
            }

            switch (event) {
                case "ORDER_CREATED":
                    handleOrderCreated(eventData);
                    break;
                case "ORDER_SCHEDULED":
                    handleOrderScheduled(eventData);
                    break;
                case "PRODUCTION_STARTED":
                    handleProductionStarted(eventData);
                    break;
                case "PRODUCTION_COMPLETED":
                    handleProductionCompleted(eventData);
                    break;
                case "ORDER_CANCELLED":
                    handleOrderCancelled(eventData);
                    break;
                default:
                    log.debug("Ignoring production event type: {}", event);
                    break;
            }
        } catch (Exception e) {
            log.error("Failed to process production event: {}", message, e);
        }
    }

    private void handleOrderCreated(Map<String, Object> eventData) {
        String orderCode = (String) eventData.get("orderCode");
        String productCode = (String) eventData.get("productCode");
        log.info("Production order created: orderCode={}, productCode={}", orderCode, productCode);

        // Auto-create MES work order from ERP production order
        try {
            mesDomainService.createWorkOrder(
                "WO-" + orderCode,
                productCode,
                "From ERP Order: " + orderCode,
                0,
                null,
                null
            );
            log.info("Auto-created MES work order from production order: {}", orderCode);
        } catch (Exception e) {
            log.error("Failed to auto-create MES work order for production order: {}", orderCode, e);
        }
    }

    private void handleOrderScheduled(Map<String, Object> eventData) {
        String orderCode = (String) eventData.get("orderCode");
        String line = (String) eventData.get("line");
        log.info("Production order scheduled: orderCode={}, line={}", orderCode, line);
    }

    private void handleProductionStarted(Map<String, Object> eventData) {
        String orderCode = (String) eventData.get("orderCode");
        log.info("Production started: orderCode={}", orderCode);
    }

    private void handleProductionCompleted(Map<String, Object> eventData) {
        String orderCode = (String) eventData.get("orderCode");
        Object quantity = eventData.get("quantity");
        log.info("Production completed: orderCode={}, quantity={}", orderCode, quantity);
    }

    private void handleOrderCancelled(Map<String, Object> eventData) {
        String orderCode = (String) eventData.get("orderCode");
        log.info("Production order cancelled: orderCode={}", orderCode);
    }
}
