package com.metawebthree.mes.infrastructure.event;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Captor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.kafka.core.KafkaTemplate;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class MesCrossDomainEventPublisherTest {

    @Mock
    private KafkaTemplate<String, String> kafkaTemplate;
    @Mock
    private ObjectMapper objectMapper;

    @Captor
    private ArgumentCaptor<String> topicCaptor;
    @Captor
    private ArgumentCaptor<String> keyCaptor;
    @Captor
    private ArgumentCaptor<String> messageCaptor;

    private MesCrossDomainEventPublisher publisher;

    @BeforeEach
    void setUp() {
        publisher = new MesCrossDomainEventPublisher(kafkaTemplate, objectMapper);
    }

    @Test
    void publishWorkOrderCompleted_shouldSendKafkaMessage() throws Exception {
        String json = "{\"eventId\":\"uuid\",\"eventType\":\"MES_WORK_ORDER_COMPLETED\"}";
        when(objectMapper.writeValueAsString(any())).thenReturn(json);

        publisher.publishWorkOrderCompleted(1L, "WO-001", "P001", 100);

        verify(objectMapper).writeValueAsString(any());
        verify(kafkaTemplate).send(topicCaptor.capture(), keyCaptor.capture(), messageCaptor.capture());
        assertEquals("mes.work_order_completed", topicCaptor.getValue());
        assertEquals("1", keyCaptor.getValue());
        assertEquals(json, messageCaptor.getValue());
    }

    @Test
    void publishTaskCompleted_shouldSendKafkaMessage() throws Exception {
        String json = "{\"eventId\":\"uuid\",\"eventType\":\"MES_TASK_COMPLETED\"}";
        when(objectMapper.writeValueAsString(any())).thenReturn(json);

        publisher.publishTaskCompleted(2L, "TASK-001", 1L, "WO-001", "P001", 90, 10);

        verify(objectMapper).writeValueAsString(any());
        verify(kafkaTemplate).send(topicCaptor.capture(), keyCaptor.capture(), messageCaptor.capture());
        assertEquals("mes.task_completed", topicCaptor.getValue());
        assertEquals("2", keyCaptor.getValue());
        assertEquals(json, messageCaptor.getValue());
    }

    @Test
    void publish_whenJsonProcessingException_shouldThrowIllegalStateException() throws Exception {
        when(objectMapper.writeValueAsString(any()))
                .thenThrow(new JsonProcessingException("Serialization error") {});

        assertThrows(IllegalStateException.class,
                () -> publisher.publishWorkOrderCompleted(1L, "WO-001", "P001", 100));

        verify(objectMapper).writeValueAsString(any());
        verify(kafkaTemplate, never()).send(anyString(), anyString(), anyString());
    }
}
