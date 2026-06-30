package com.metawebthree.digitaltwin.kafka;

import com.metawebthree.digitaltwin.application.ai.WarehouseAIService;
import com.metawebthree.digitaltwin.interfaces.websocket.DigitalTwinWebSocketHandler;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.lang.reflect.Method;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class DigitalTwinKafkaConsumerTest {

    @Mock
    private DigitalTwinWebSocketHandler webSocketHandler;
    @Mock
    private WarehouseAIService aiService;

    private DigitalTwinKafkaConsumer consumer;

    @BeforeEach
    void setUp() {
        consumer = new DigitalTwinKafkaConsumer(webSocketHandler, aiService);
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> mapMatcher(String expectedType) {
        return argThat((Map<String, Object> map) -> expectedType.equals(map.get("type")));
    }

    @Test
    void consumeDeviceStatusChanged_shouldBroadcastMessage() {
        String message = "{\"messageId\":\"msg-001\",\"deviceCode\":\"DEV-001\",\"status\":\"ONLINE\"}";
        
        consumer.consumeDeviceStatusChanged(message);

        verify(webSocketHandler).broadcast(mapMatcher("DEVICE_STATUS_CHANGED"));
    }

    @Test
    void consumeDevicePositionUpdated_shouldBroadcastMessage() {
        String message = "{\"messageId\":\"msg-002\",\"deviceCode\":\"DEV-001\",\"x\":10.0,\"y\":20.0}";
        
        consumer.consumeDevicePositionUpdated(message);

        verify(webSocketHandler).broadcast(mapMatcher("DEVICE_POSITION_UPDATED"));
    }

    @Test
    void consumeDeviceHeartbeat_shouldNotBroadcast() {
        String message = "{\"messageId\":\"msg-003\",\"deviceCode\":\"DEV-001\"}";
        
        consumer.consumeDeviceHeartbeat(message);

        verify(webSocketHandler, never()).broadcast(anyMap());
    }

    @Test
    void consumeAlertCreated_shouldBroadcastMessage() {
        String message = "{\"messageId\":\"msg-004\",\"alertCode\":\"ALT-001\",\"level\":\"ERROR\"}";
        
        consumer.consumeAlertCreated(message);

        verify(webSocketHandler).broadcast(mapMatcher("ALERT_CREATED"));
    }

    @Test
    void consumeProductionOutputUpdated_shouldBroadcastMessage() {
        String message = "{\"messageId\":\"msg-005\",\"lineCode\":\"PL001\",\"output\":50}";
        
        consumer.consumeProductionOutputUpdated(message);

        verify(webSocketHandler).broadcast(mapMatcher("PRODUCTION_OUTPUT_UPDATED"));
    }

    @Test
    void consumeAgvPositionUpdated_shouldBroadcastMessage() {
        String message = "{\"messageId\":\"msg-006\",\"agvCode\":\"AGV-001\",\"x\":5.0,\"y\":10.0}";
        
        consumer.consumeAgvPositionUpdated(message);

        verify(webSocketHandler).broadcast(mapMatcher("AGV_POSITION_UPDATED"));
    }

    @Test
    void processMessage_shouldHandleDuplicateMessage() {
        String message = "{\"messageId\":\"dup-001\",\"deviceCode\":\"DEV-001\"}";
        
        consumer.consumeDeviceStatusChanged(message);
        consumer.consumeDeviceStatusChanged(message);

        verify(webSocketHandler, times(1)).broadcast(anyMap());
    }

    @Test
    void processMessage_shouldHandleMessageWithoutMessageId() {
        String message = "{\"deviceCode\":\"DEV-001\",\"timestamp\":1234567890}";
        
        consumer.consumeDeviceStatusChanged(message);

        verify(webSocketHandler).broadcast(anyMap());
    }

    @Test
    void processMessage_shouldHandleMessageWithDeviceCodeAndTimestamp() {
        String message = "{\"deviceCode\":\"DEV-001\",\"timestamp\":1234567890}";
        
        consumer.consumeDevicePositionUpdated(message);

        verify(webSocketHandler).broadcast(anyMap());
    }

    @Test
    void processMessage_shouldHandleInvalidJson() {
        String invalidMessage = "{invalid json";
        
        consumer.consumeDeviceStatusChanged(invalidMessage);

        verify(webSocketHandler).broadcast(anyMap());
    }

    @Test
    void isDuplicate_shouldReturnFalseForNewMessageId() {
        String messageId = "unique-001";
        
        boolean result = invokeIsDuplicate(messageId);
        
        assertFalse(result);
    }

    @Test
    void isDuplicate_shouldReturnTrueForExistingMessageId() {
        String messageId = "dup-test-001";
        
        invokeIsDuplicate(messageId);
        boolean result = invokeIsDuplicate(messageId);
        
        assertTrue(result);
    }

    private boolean invokeIsDuplicate(String messageId) {
        try {
            Method method = DigitalTwinKafkaConsumer.class.getDeclaredMethod("isDuplicate", String.class);
            method.setAccessible(true);
            return (boolean) method.invoke(consumer, messageId);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}