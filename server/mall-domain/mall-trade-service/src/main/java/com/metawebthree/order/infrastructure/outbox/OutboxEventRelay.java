package com.metawebthree.order.infrastructure.outbox;

import java.time.LocalDateTime;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import com.metawebthree.order.domain.model.OutboxEventDO;
import com.metawebthree.order.infrastructure.persistence.mapper.OutboxEventMapper;

import lombok.extern.slf4j.Slf4j;

/**
 * Polls the durable outbox and forwards rows to Kafka at least once.
 *
 * <p>Delivery semantics: a row is atomically claimed ({@code PENDING -> SENDING})
 * before publishing, so concurrent relay instances never deliver the same row
 * twice at the same time. If publishing fails the row is scheduled for retry
 * until {@code max-attempts} is reached, after which it is parked as
 * {@code FAILED} for reconciliation. Because a crash can happen after a
 * successful broker write but before the row is marked {@code SENT}, consumers
 * must de-duplicate on the event id.</p>
 */
@Slf4j
@Component
public class OutboxEventRelay {

    private final OutboxEventMapper outboxEventMapper;
    private final KafkaTemplate<String, String> kafkaTemplate;

    @Value("${event.outbox.batch-size:100}")
    private int batchSize;

    @Value("${event.outbox.max-attempts:10}")
    private int maxAttempts;

    @Value("${event.outbox.retry-backoff-seconds:30}")
    private long retryBackoffSeconds;

    @Value("${event.outbox.send-timeout-seconds:10}")
    private long sendTimeoutSeconds;

    public OutboxEventRelay(OutboxEventMapper outboxEventMapper,
                            KafkaTemplate<String, String> kafkaTemplate) {
        this.outboxEventMapper = outboxEventMapper;
        this.kafkaTemplate = kafkaTemplate;
    }

    @Scheduled(fixedDelayString = "${event.outbox.poll-interval-ms:5000}")
    public void relayPendingEvents() {
        List<OutboxEventDO> pending = outboxEventMapper.findPending(LocalDateTime.now(), batchSize);
        for (OutboxEventDO event : pending) {
            try {
                deliver(event);
            } catch (Exception e) {
                log.error("Unexpected failure while delivering outbox event id={}", event.getId(), e);
            }
        }
    }

    private void deliver(OutboxEventDO event) throws Exception {
        LocalDateTime now = LocalDateTime.now();
        if (outboxEventMapper.claimPending(event.getId(), now) == 0) {
            // Another relay instance owns this row.
            return;
        }
        try {
            kafkaTemplate.send(event.getTopic(), event.getEventKey(), event.getPayload())
                    .get(sendTimeoutSeconds, TimeUnit.SECONDS);
            outboxEventMapper.markSent(event.getId(), LocalDateTime.now());
            log.debug("Outbox event delivered: id={}, topic={}, eventId={}",
                    event.getId(), event.getTopic(), event.getEventId());
        } catch (Exception e) {
            int attempts = event.getAttempts() == null ? 1 : event.getAttempts() + 1;
            boolean exhausted = attempts >= maxAttempts;
            String status = exhausted ? OutboxEventDO.STATUS_FAILED : OutboxEventDO.STATUS_PENDING;
            LocalDateTime nextAttemptAt = exhausted
                    ? null
                    : LocalDateTime.now().plusSeconds(retryBackoffSeconds);
            outboxEventMapper.markAttemptFailed(event.getId(), status, truncate(e.getMessage()),
                    nextAttemptAt, LocalDateTime.now());
            if (exhausted) {
                log.error("Outbox event parked as FAILED after {} attempts: id={}, topic={}, eventId={}",
                        attempts, event.getId(), event.getTopic(), event.getEventId(), e);
            } else {
                log.warn("Outbox delivery failed (attempt {}): id={}, will retry at {}",
                        attempts, event.getId(), nextAttemptAt, e);
            }
        }
    }

    private String truncate(String message) {
        if (message == null) {
            return null;
        }
        return message.length() > 900 ? message.substring(0, 900) : message;
    }
}
