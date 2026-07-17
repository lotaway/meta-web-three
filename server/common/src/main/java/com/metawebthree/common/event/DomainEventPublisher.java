package com.metawebthree.common.event;

import java.util.Map;

public interface DomainEventPublisher {

    void publish(String eventType, Map<String, Object> data);
}