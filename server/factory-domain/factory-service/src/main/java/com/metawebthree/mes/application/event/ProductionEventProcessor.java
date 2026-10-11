package com.metawebthree.mes.application.event;

import java.util.Map;

public interface ProductionEventProcessor {
    void handleOrderCreated(Map<String, Object> eventData);
    void handleOrderCancelled(Map<String, Object> eventData);
}
