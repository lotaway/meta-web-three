package com.metawebthree.order.infrastructure.outbox;

import com.metawebthree.order.domain.model.OutboxEventDO;
import com.metawebthree.order.infrastructure.persistence.mapper.OutboxEventMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.support.SendResult;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.concurrent.CompletableFuture;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class OutboxEventRelayTest {

    private OutboxEventMapper outboxEventMapper;
    private KafkaTemplate<String, String> kafkaTemplate;
    private OutboxEventRelay relay;

    @BeforeEach
    void setUp() {
        outboxEventMapper = mock(OutboxEventMapper.class);
        kafkaTemplate = mock(KafkaTemplate.class);
        relay = new OutboxEventRelay(outboxEventMapper, kafkaTemplate);
        org.springframework.test.util.ReflectionTestUtils.setField(relay, "batchSize", 100);
        org.springframework.test.util.ReflectionTestUtils.setField(relay, "maxAttempts", 3);
        org.springframework.test.util.ReflectionTestUtils.setField(relay, "retryBackoffSeconds", 30L);
        org.springframework.test.util.ReflectionTestUtils.setField(relay, "sendTimeoutSeconds", 1L);
    }

    private OutboxEventDO event(int attempts) {
        return OutboxEventDO.builder()
                .id(1L)
                .eventId("evt-1")
                .topic("order.created")
                .eventKey("NO-1")
                .payload("{}")
                .status(OutboxEventDO.STATUS_PENDING)
                .attempts(attempts)
                .build();
    }

    @Test
    void deliversAndMarksSent() {
        when(outboxEventMapper.findPending(any(), anyInt())).thenReturn(java.util.List.of(event(0)));
        when(outboxEventMapper.claimPending(eq(1L), any())).thenReturn(1);
        when(kafkaTemplate.send(anyString(), anyString(), anyString()))
                .thenReturn(java.util.concurrent.CompletableFuture.completedFuture(null));

        relay.relayPendingEvents();

        verify(kafkaTemplate).send("order.created", "NO-1", "{}");
        verify(outboxEventMapper).markSent(eq(1L), any());
        verify(outboxEventMapper, never()).markAttemptFailed(any(), anyString(), any(), any(), any());
    }

    @Test
    void losingClaimDoesNotPublish() {
        when(outboxEventMapper.findPending(any(), anyInt())).thenReturn(java.util.List.of(event(0)));
        when(outboxEventMapper.claimPending(eq(1L), any())).thenReturn(0);

        relay.relayPendingEvents();

        verify(kafkaTemplate, never()).send(anyString(), anyString(), anyString());
        verify(outboxEventMapper, never()).markSent(any(), any());
    }

    @Test
    void failureIsRescheduledForRetry() {
        when(outboxEventMapper.findPending(any(), anyInt())).thenReturn(java.util.List.of(event(0)));
        when(outboxEventMapper.claimPending(eq(1L), any())).thenReturn(1);
        when(kafkaTemplate.send(anyString(), anyString(), anyString()))
                .thenReturn(java.util.concurrent.CompletableFuture.failedFuture(new RuntimeException("broker down")));

        relay.relayPendingEvents();

        verify(outboxEventMapper).markAttemptFailed(eq(1L), eq(OutboxEventDO.STATUS_PENDING),
                any(), any(), any());
        verify(outboxEventMapper, never()).markSent(any(), any());
    }

    @Test
    void failureAfterMaxAttemptsIsParked() {
        when(outboxEventMapper.findPending(any(), anyInt())).thenReturn(java.util.List.of(event(2)));
        when(outboxEventMapper.claimPending(eq(1L), any())).thenReturn(1);
        when(kafkaTemplate.send(anyString(), anyString(), anyString()))
                .thenReturn(java.util.concurrent.CompletableFuture.failedFuture(new RuntimeException("broker down")));

        relay.relayPendingEvents();

        verify(outboxEventMapper).markAttemptFailed(eq(1L), eq(OutboxEventDO.STATUS_FAILED),
                any(), isNull(), any());
    }
}
