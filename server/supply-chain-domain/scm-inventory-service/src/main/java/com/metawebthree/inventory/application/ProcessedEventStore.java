package com.metawebthree.inventory.application;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.inventory.domain.entity.ProcessedEvent;
import com.metawebthree.inventory.infrastructure.persistence.mapper.ProcessedEventMapper;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;

/**
 * Consumer side de-duplication keyed by the event id. Because the producer only
 * guarantees at-least-once delivery, every consumer must ignore events it has
 * already handled.
 */
@Component
public class ProcessedEventStore {

    private final ProcessedEventMapper processedEventMapper;

    public ProcessedEventStore(ProcessedEventMapper processedEventMapper) {
        this.processedEventMapper = processedEventMapper;
    }

    /**
     * @return true when this event id has not been handled by the given consumer
     *         yet; false when it is a redelivery that must be skipped
     */
    public boolean markFirstProcessing(String eventId, String consumerGroup) {
        if (eventId == null || eventId.isBlank()) {
            // Events without an id cannot be de-duplicated; process them so that
            // valid events are never dropped.
            return true;
        }
        if (processedEventMapper.countByEvent(eventId, consumerGroup) > 0) {
            return false;
        }
        ProcessedEvent record = new ProcessedEvent();
        record.setId(IdWorker.getId());
        record.setEventId(eventId);
        record.setConsumerGroup(consumerGroup);
        record.setProcessedAt(LocalDateTime.now());
        try {
            processedEventMapper.insert(record);
            return true;
        } catch (DuplicateKeyException e) {
            // Concurrent handling of the same event won the insert race.
            return false;
        }
    }
}
