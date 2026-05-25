package com.metawebthree.digitaltwin.infrastructure.event;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.support.SendResult;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.CompletableFuture;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class DigitalTwinEventPublisherTest {

    @Mock
    private KafkaTemplate<String, String> kafkaTemplate;

    @Mock
    private SendResult<String, String> sendResult;

    private DigitalTwinEventPublisher publisher;

    @BeforeEach
    void setUp() {
        publisher = new DigitalTwinEventPublisher(kafkaTemplate);
        ReflectionTestUtils.setField(publisher, "topicPrefix", "digital-twin");
    }

    @Test
    void publishDeviceRegistered_shouldSendToKafka() {
        when(kafkaTemplate.send(anyString(), anyString())).thenReturn(CompletableFuture.completedFuture(sendResult));

        publisher.publishDeviceRegistered("DEVICE-001");

        verify(kafkaTemplate).send(eq("digital-twin.device.registered"), anyString());
    }

    @Test
    void publishDeviceStatusChanged_shouldSendToKafka() {
        when(kafkaTemplate.send(anyString(), anyString())).thenReturn(CompletableFuture.completedFuture(sendResult));

        publisher.publishDeviceStatusChanged("DEVICE-001", "ONLINE");

        verify(kafkaTemplate).send(eq("digital-twin.device.status.changed"), anyString());
    }

    @Test
    void publishDevicePositionUpdated_shouldSendToKafka() {
        when(kafkaTemplate.send(anyString(), anyString())).thenReturn(CompletableFuture.completedFuture(sendResult));

        publisher.publishDevicePositionUpdated("DEVICE-001", 10.0, 20.0, 30.0);

        verify(kafkaTemplate).send(eq("digital-twin.device.position.updated"), anyString());
    }

    @Test
    void publishWorkshopCreated_shouldSendToKafka() {
        when(kafkaTemplate.send(anyString(), anyString())).thenReturn(CompletableFuture.completedFuture(sendResult));

        publisher.publishWorkshopCreated("WORKSHOP-001");

        verify(kafkaTemplate).send(eq("digital-twin.workshop.created"), anyString());
    }

    @Test
    void publishProductionLineCreated_shouldSendToKafka() {
        when(kafkaTemplate.send(anyString(), anyString())).thenReturn(CompletableFuture.completedFuture(sendResult));

        publisher.publishProductionLineCreated("LINE-001");

        verify(kafkaTemplate).send(eq("digital-twin.production.line.created"), anyString());
    }

    @Test
    void publishProductionOutputUpdated_shouldSendToKafka() {
        when(kafkaTemplate.send(anyString(), anyString())).thenReturn(CompletableFuture.completedFuture(sendResult));

        publisher.publishProductionOutputUpdated("LINE-001", 100);

        verify(kafkaTemplate).send(eq("digital-twin.production.output.updated"), anyString());
    }

    @Test
    void publishAlertCreated_shouldSendToKafka() {
        when(kafkaTemplate.send(anyString(), anyString())).thenReturn(CompletableFuture.completedFuture(sendResult));

        publisher.publishAlertCreated("ALERT-001", "WARNING");

        verify(kafkaTemplate).send(eq("digital-twin.alert.created"), anyString());
    }

    @Test
    void publishAlertAcknowledged_shouldSendToKafka() {
        when(kafkaTemplate.send(anyString(), anyString())).thenReturn(CompletableFuture.completedFuture(sendResult));

        publisher.publishAlertAcknowledged(1L);

        verify(kafkaTemplate).send(eq("digital-twin.alert.acknowledged"), anyString());
    }

    @Test
    void publishAlertResolved_shouldSendToKafka() {
        when(kafkaTemplate.send(anyString(), anyString())).thenReturn(CompletableFuture.completedFuture(sendResult));

        publisher.publishAlertResolved(1L);

        verify(kafkaTemplate).send(eq("digital-twin.alert.resolved"), anyString());
    }

    @Test
    void publishWarehouseStatusChanged_shouldSendToKafka() {
        when(kafkaTemplate.send(anyString(), anyString())).thenReturn(CompletableFuture.completedFuture(sendResult));

        publisher.publishWarehouseStatusChanged("WH-001", "OPERATING");

        verify(kafkaTemplate).send(eq("digital-twin.warehouse.status.changed"), anyString());
    }

    @Test
    void publishInventoryLevelChanged_shouldSendToKafka() {
        when(kafkaTemplate.send(anyString(), anyString())).thenReturn(CompletableFuture.completedFuture(sendResult));

        publisher.publishInventoryLevelChanged("WH-001", "SKU-001", 100, "NORMAL");

        verify(kafkaTemplate).send(eq("digital-twin.inventory.level.changed"), anyString());
    }

    @Test
    void publishInventoryAlertCreated_shouldSendToKafka() {
        when(kafkaTemplate.send(anyString(), anyString())).thenReturn(CompletableFuture.completedFuture(sendResult));

        publisher.publishInventoryAlertCreated("WH-001", "ALERT-001", "WARNING", "Low stock");

        verify(kafkaTemplate).send(eq("digital-twin.inventory.alert.created"), anyString());
    }

    @Test
    void publishRestockSuggestionCreated_shouldSendToKafka() {
        when(kafkaTemplate.send(anyString(), anyString())).thenReturn(CompletableFuture.completedFuture(sendResult));

        publisher.publishRestockSuggestionCreated("WH-001", "SKU-001", 50, "Below minimum");

        verify(kafkaTemplate).send(eq("digital-twin.restock.suggestion.created"), anyString());
    }

    @Test
    void publishShelfStatusChanged_shouldSendToKafka() {
        when(kafkaTemplate.send(anyString(), anyString())).thenReturn(CompletableFuture.completedFuture(sendResult));

        publisher.publishShelfStatusChanged("WH-001", "SHELF-001", "OCCUPIED");

        verify(kafkaTemplate).send(eq("digital-twin.shelf.status.changed"), anyString());
    }

    @Test
    void publishToKafka_shouldHandleFailure() {
        when(kafkaTemplate.send(anyString(), anyString())).thenReturn(CompletableFuture.failedFuture(new RuntimeException("Kafka error")));

        publisher.publishDeviceRegistered("DEVICE-001");

        verify(kafkaTemplate).send(anyString(), anyString());
    }
}