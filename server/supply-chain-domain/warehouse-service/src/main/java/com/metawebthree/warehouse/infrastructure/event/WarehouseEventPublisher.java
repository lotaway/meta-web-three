package com.metawebthree.warehouse.infrastructure.event;

import com.metawebthree.event.EventPublisher;
import com.metawebthree.event.EventType;
import org.springframework.stereotype.Component;
import java.util.HashMap;
import java.util.Map;

@Component
public class WarehouseEventPublisher {

    private final EventPublisher eventPublisher;

    public WarehouseEventPublisher(EventPublisher eventPublisher) {
        this.eventPublisher = eventPublisher;
    }

    public void publishCreated(Long warehouseId, String warehouseCode, String name) {
        Map<String, Object> data = new HashMap<>();
        data.put("warehouseId", warehouseId);
        data.put("warehouseCode", warehouseCode);
        data.put("name", name);
        eventPublisher.publish(EventType.WAREHOUSE_CREATED, data);
    }

    public void publishStockIn(Long warehouseId, String skuCode, Integer quantity, String orderNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("warehouseId", warehouseId);
        data.put("skuCode", skuCode);
        data.put("quantity", quantity);
        data.put("orderNo", orderNo);
        eventPublisher.publish(EventType.WAREHOUSE_STOCK_IN, data);
    }

    public void publishStockOut(Long warehouseId, String skuCode, Integer quantity, String orderNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("warehouseId", warehouseId);
        data.put("skuCode", skuCode);
        data.put("quantity", quantity);
        data.put("orderNo", orderNo);
        eventPublisher.publish(EventType.WAREHOUSE_STOCK_OUT, data);
    }

    public void publishInboundOrderCreated(String orderNo, Long warehouseId) {
        Map<String, Object> data = new HashMap<>();
        data.put("orderNo", orderNo);
        data.put("warehouseId", warehouseId);
        eventPublisher.publish(EventType.INBOUND_ORDER_CREATED, data);
    }

    public void publishInboundOrderCompleted(String orderNo) {
        Map<String, Object> data = new HashMap<>();
        data.put("orderNo", orderNo);
        eventPublisher.publish(EventType.INBOUND_ORDER_COMPLETED, data);
    }
}