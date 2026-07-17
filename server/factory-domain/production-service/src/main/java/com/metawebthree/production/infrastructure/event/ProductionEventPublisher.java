package com.metawebthree.production.infrastructure.event;

import com.metawebthree.event.EventType;
import com.metawebthree.production.domain.entity.ProductionOrder;
import com.metawebthree.production.domain.entity.WorkStation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;

@Component
public class ProductionEventPublisher {
    private static final Logger logger = LoggerFactory.getLogger(ProductionEventPublisher.class);
    
    private final KafkaTemplate<String, String> kafkaTemplate;

    public ProductionEventPublisher(KafkaTemplate<String, String> kafkaTemplate) {
        this.kafkaTemplate = kafkaTemplate;
    }

    public void publishOrderCreated(ProductionOrder order) {
        String message = String.format("{\"event\":\"%s\",\"orderCode\":\"%s\",\"productCode\":\"%s\"}",
            EventType.ORDER_CREATED.name(), order.getOrderCode(), order.getProductCode());
        kafkaTemplate.send(EventType.PRODUCTION_EVENTS_TOPIC, message);
        logger.info("Published ORDER_CREATED event for: {}", order.getOrderCode());
    }

    public void publishOrderScheduled(ProductionOrder order) {
        String message = String.format("{\"event\":\"ORDER_SCHEDULED\",\"orderCode\":\"%s\",\"line\":\"%s\"}",
            order.getOrderCode(), order.getProductionLineCode());
        kafkaTemplate.send(EventType.PRODUCTION_EVENTS_TOPIC, message);
        logger.info("Published ORDER_SCHEDULED event for: {}", order.getOrderCode());
    }

    public void publishProductionStarted(ProductionOrder order) {
        String message = String.format("{\"event\":\"PRODUCTION_STARTED\",\"orderCode\":\"%s\"}", order.getOrderCode());
        kafkaTemplate.send(EventType.PRODUCTION_EVENTS_TOPIC, message);
        logger.info("Published PRODUCTION_STARTED event for: {}", order.getOrderCode());
    }

    public void publishProductionPaused(ProductionOrder order) {
        String message = String.format("{\"event\":\"PRODUCTION_PAUSED\",\"orderCode\":\"%s\"}", order.getOrderCode());
        kafkaTemplate.send(EventType.PRODUCTION_EVENTS_TOPIC, message);
        logger.info("Published PRODUCTION_PAUSED event for: {}", order.getOrderCode());
    }

    public void publishProductionResumed(ProductionOrder order) {
        String message = String.format("{\"event\":\"PRODUCTION_RESUMED\",\"orderCode\":\"%s\"}", order.getOrderCode());
        kafkaTemplate.send(EventType.PRODUCTION_EVENTS_TOPIC, message);
        logger.info("Published PRODUCTION_RESUMED event for: {}", order.getOrderCode());
    }

    public void publishProductionCompleted(ProductionOrder order) {
        String message = String.format("{\"event\":\"PRODUCTION_COMPLETED\",\"orderCode\":\"%s\",\"quantity\":%d}",
            order.getOrderCode(), order.getQuantityCompleted());
        kafkaTemplate.send(EventType.PRODUCTION_EVENTS_TOPIC, message);
        logger.info("Published PRODUCTION_COMPLETED event for: {}", order.getOrderCode());
    }

    public void publishOrderCancelled(ProductionOrder order) {
        String message = String.format("{\"event\":\"%s\",\"orderCode\":\"%s\"}",
            EventType.ORDER_CANCELLED.name(), order.getOrderCode());
        kafkaTemplate.send(EventType.PRODUCTION_EVENTS_TOPIC, message);
        logger.info("Published ORDER_CANCELLED event for: {}", order.getOrderCode());
    }

    public void publishWorkStationCreated(WorkStation station) {
        String message = String.format("{\"event\":\"STATION_CREATED\",\"stationCode\":\"%s\",\"type\":\"%s\"}",
            station.getStationCode(), station.getStationType());
        kafkaTemplate.send(EventType.PRODUCTION_EVENTS_TOPIC, message);
        logger.info("Published STATION_CREATED event for: {}", station.getStationCode());
    }

    public void publishOrderAssignedToStation(WorkStation station, String orderCode) {
        String message = String.format("{\"event\":\"ORDER_ASSIGNED_TO_STATION\",\"stationCode\":\"%s\",\"orderCode\":\"%s\"}",
            station.getStationCode(), orderCode);
        kafkaTemplate.send(EventType.PRODUCTION_EVENTS_TOPIC, message);
        logger.info("Published ORDER_ASSIGNED_TO_STATION event: {} -> {}", orderCode, station.getStationCode());
    }
}