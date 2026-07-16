package com.metawebthree.mes.application.event;

public interface CrossDomainEventPublisher {
    void publishWorkOrderCompleted(Long workOrderId, String workOrderNo, String productCode, Integer quantity);
    void publishTaskCompleted(Long taskId, String taskNo, Long workOrderId, String workOrderNo, String productCode, Integer qualified, Integer defective);
}
