package com.metawebthree.mes.infrastructure.event;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.event.EventType;
import com.metawebthree.mes.application.event.ProductionEventProcessor;
import com.metawebthree.mes.domain.service.MesDomainService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

import java.util.Map;

@Component
public class ProductionOrderEventListener implements ProductionEventProcessor {

    private static final Logger log = LoggerFactory.getLogger(ProductionOrderEventListener.class);

    private final MesDomainService mesDomainService;
    private final ObjectMapper objectMapper;

    public ProductionOrderEventListener(MesDomainService mesDomainService,
                                        ObjectMapper objectMapper) {
        this.mesDomainService = mesDomainService;
        this.objectMapper = objectMapper;
    }

    @KafkaListener(topics = EventType.PRODUCTION_EVENTS_TOPIC, groupId = "mes-service")
    public void onProductionEvent(String message) {
        try {
            Map<String, Object> eventData = objectMapper.readValue(message, Map.class);
            String event = (String) eventData.get("event");
            if (event == null) {
                log.warn("Unknown production event format: {}", message);
                return;
            }
            dispatchEvent(event, eventData);
        } catch (Exception e) {
            log.error("Failed to process production event: {}", message, e);
        }
    }

    private void dispatchEvent(String event, Map<String, Object> eventData) {
        if (EventType.ORDER_CREATED.name().equals(event)) {
            handleOrderCreated(eventData);
        } else if (EventType.ORDER_CANCELLED.name().equals(event)) {
            handleOrderCancelled(eventData);
        } else {
            log.debug("Ignoring production event type: {}", event);
        }
    }

    @Override
    public void handleOrderCreated(Map<String, Object> eventData) {
        String orderCode = (String) eventData.get("orderCode");
        String productCode = (String) eventData.get("productCode");
        try {
            mesDomainService.createWorkOrder(
                "WO-" + orderCode,
                productCode,
                "From ERP Order: " + orderCode,
                0, null, null
            );
            log.info("Auto-created MES work order from production order: {}", orderCode);
        } catch (Exception e) {
            log.error("Failed to auto-create MES work order for production order: {}", orderCode, e);
        }
    }

    @Override
    public void handleOrderCancelled(Map<String, Object> eventData) {
        String orderCode = (String) eventData.get("orderCode");
        log.info("Production order cancelled: orderCode={}", orderCode);
    }
}
