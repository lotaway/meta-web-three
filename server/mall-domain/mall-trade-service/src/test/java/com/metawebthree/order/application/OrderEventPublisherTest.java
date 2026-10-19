package com.metawebthree.order.application;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.order.domain.model.OutboxEventDO;
import com.metawebthree.order.infrastructure.persistence.mapper.OutboxEventMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.test.util.ReflectionTestUtils;

import java.math.BigDecimal;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

class OrderEventPublisherTest {

    private OutboxEventMapper outboxEventMapper;
    private OrderEventPublisher publisher;

    @BeforeEach
    void setUp() {
        outboxEventMapper = mock(OutboxEventMapper.class);
        publisher = new OrderEventPublisher(outboxEventMapper, new ObjectMapper().findAndRegisterModules());
        ReflectionTestUtils.setField(publisher, "orderCreatedTopic", "order.created");
    }

    @Test
    void publishOrderCreatedPersistsOutboxRow() {
        List<OrderEventPublisher.OrderItemCreate> items = List.of(
                new OrderEventPublisher.OrderItemCreate(7L, "Widget", 9L, 2,
                        new BigDecimal("10.00"), "img.png"));

        publisher.publishOrderCreated(100L, "NO-100", 5L, new BigDecimal("20.00"), "USD", items);

        ArgumentCaptor<OutboxEventDO> captor = ArgumentCaptor.forClass(OutboxEventDO.class);
        verify(outboxEventMapper).insert(captor.capture());
        OutboxEventDO row = captor.getValue();
        assertEquals("order.created", row.getTopic());
        assertEquals("ORDER", row.getAggregateType());
        assertEquals("100", row.getAggregateId());
        assertEquals("ORDER_CREATED", row.getEventType());
        assertEquals(OutboxEventDO.STATUS_PENDING, row.getStatus());
        assertEquals(0, row.getAttempts());
        assertEquals("NO-100", row.getEventKey());
        assertNotNull(row.getId());
        assertNotNull(row.getEventId());
        assertTrue(row.getPayload().contains("\"orderId\":\"100\""), row.getPayload());
        assertTrue(row.getPayload().contains("\"eventId\""), row.getPayload());
    }
}
