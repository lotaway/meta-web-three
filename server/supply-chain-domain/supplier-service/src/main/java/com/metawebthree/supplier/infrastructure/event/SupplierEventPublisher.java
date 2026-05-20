package com.metawebthree.supplier.infrastructure.event;

import com.metawebthree.event.EventPublisher;
import com.metawebthree.event.EventType;
import org.springframework.stereotype.Component;
import java.util.HashMap;
import java.util.Map;

@Component
public class SupplierEventPublisher {

    private final EventPublisher eventPublisher;

    public SupplierEventPublisher(EventPublisher eventPublisher) {
        this.eventPublisher = eventPublisher;
    }

    public void publishCreated(Long id, String supplierCode, String name) {
        Map<String, Object> data = new HashMap<>();
        data.put("supplierId", id);
        data.put("supplierCode", supplierCode);
        data.put("name", name);
        eventPublisher.publish(EventType.SUPPLIER_CREATED, data);
    }

    public void publishUpdated(Long id, String supplierCode) {
        Map<String, Object> data = new HashMap<>();
        data.put("supplierId", id);
        data.put("supplierCode", supplierCode);
        eventPublisher.publish(EventType.SUPPLIER_UPDATED, data);
    }

    public void publishAssessmentChanged(Long id, String supplierCode, String level) {
        Map<String, Object> data = new HashMap<>();
        data.put("supplierId", id);
        data.put("supplierCode", supplierCode);
        data.put("assessmentLevel", level);
        eventPublisher.publish(EventType.SUPPLIER_ASSESSMENT_CHANGED, data);
    }

    public void publishStatusChanged(Long id, String supplierCode, String status) {
        Map<String, Object> data = new HashMap<>();
        data.put("supplierId", id);
        data.put("supplierCode", supplierCode);
        data.put("status", status);
        eventPublisher.publish(EventType.SUPPLIER_STATUS_CHANGED, data);
    }
}