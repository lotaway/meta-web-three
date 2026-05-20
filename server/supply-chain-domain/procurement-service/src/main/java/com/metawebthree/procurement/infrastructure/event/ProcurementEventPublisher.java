package com.metawebthree.procurement.infrastructure.event;

import com.metawebthree.event.EventPublisher;
import com.metawebthree.event.EventType;
import org.springframework.stereotype.Component;
import java.util.HashMap;
import java.util.Map;

@Component
public class ProcurementEventPublisher {

    private final EventPublisher eventPublisher;

    public ProcurementEventPublisher(EventPublisher eventPublisher) {
        this.eventPublisher = eventPublisher;
    }

    public void publishCreated(String orderNo, String supplierCode) {
        Map<String, Object> data = new HashMap<>();
        data.put("orderNo", orderNo);
        data.put("supplierCode", supplierCode);
        eventPublisher.publish(EventType.PROCUREMENT_CREATED, data);
    }

    public void publishApproved(String orderNo, String approver) {
        Map<String, Object> data = new HashMap<>();
        data.put("orderNo", orderNo);
        data.put("approver", approver);
        eventPublisher.publish(EventType.PROCUREMENT_APPROVED, data);
    }

    public void publishRejected(String orderNo, String reason) {
        Map<String, Object> data = new HashMap<>();
        data.put("orderNo", orderNo);
        data.put("reason", reason);
        eventPublisher.publish(EventType.PROCUREMENT_REJECTED, data);
    }

    public void publishCompleted(String orderNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("orderNo", orderNo);
        eventPublisher.publish(EventType.PROCUREMENT_COMPLETED, data);
    }

    public void publishCancelled(String orderNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("orderNo", orderNo);
        eventPublisher.publish(EventType.PROCUREMENT_CANCELLED, data);
    }
}