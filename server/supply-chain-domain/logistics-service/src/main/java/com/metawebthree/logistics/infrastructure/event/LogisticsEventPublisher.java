package com.metawebthree.logistics.infrastructure.event;

import com.metawebthree.event.EventPublisher;
import com.metawebthree.event.EventType;
import org.springframework.stereotype.Component;
import java.util.HashMap;
import java.util.Map;

@Component
public class LogisticsEventPublisher {

    private final EventPublisher eventPublisher;

    public LogisticsEventPublisher(EventPublisher eventPublisher) {
        this.eventPublisher = eventPublisher;
    }

    public void publishCreated(String trackingNo, String orderNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("trackingNo", trackingNo);
        data.put("orderNo", orderNo);
        eventPublisher.publish(EventType.LOGISTICS_CREATED, data);
    }

    public void publishTrackingUpdated(String trackingNo, String status, String location) {
        Map<String, Object> data = new HashMap<>();
        data.put("trackingNo", trackingNo);
        data.put("status", status);
        data.put("location", location);
        eventPublisher.publish(EventType.LOGISTICS_TRACKING_UPDATED, data);
    }

    public void publishDispatched(String trackingNo, String carrier, String carrierOrderNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("trackingNo", trackingNo);
        data.put("carrier", carrier);
        data.put("carrierOrderNo", carrierOrderNo);
        eventPublisher.publish(EventType.LOGISTICS_DISPATCHED, data);
    }

    public void publishDelivered(String trackingNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("trackingNo", trackingNo);
        eventPublisher.publish(EventType.LOGISTICS_DELIVERED, data);
    }
}