package com.metawebthree.digitaltwin.infrastructure.event;

import org.junit.jupiter.api.Test;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.support.SendResult;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class DigitalTwinEventPublisherTest {

    @SuppressWarnings("unchecked")
    private final KafkaTemplate<String, String> kafkaTemplate = mock(KafkaTemplate.class);

    private final DigitalTwinEventPublisher publisher = new DigitalTwinEventPublisher(kafkaTemplate);

    private static CompletableFuture<SendResult<String, String>> completed() {
        return CompletableFuture.completedFuture(null);
    }

    @Test
    void publishDoesNotPropagateSynchronousSendFailure() {
        when(kafkaTemplate.send(anyString(), anyString()))
                .thenThrow(new RuntimeException("broker unavailable"));

        assertDoesNotThrow(() -> publisher.publishDeviceRegistered("AGV-001"));
    }

    @Test
    void publishDoesNotPropagateFailedFuture() {
        CompletableFuture<SendResult<String, String>> failed = new CompletableFuture<>();
        failed.completeExceptionally(new RuntimeException("send timeout"));
        when(kafkaTemplate.send(anyString(), anyString())).thenReturn(failed);

        assertDoesNotThrow(() -> publisher.publishAlertCreated("ALERT-001", "CRITICAL"));
    }

    @Test
    void publishSucceedsWithHealthyTemplate() {
        when(kafkaTemplate.send(anyString(), anyString())).thenReturn(completed());

        assertDoesNotThrow(() -> publisher.publishWorkshopCreated("WS-01"));
    }
}
