package com.metawebthree.inventory.application;

import com.metawebthree.inventory.domain.entity.ProcessedEvent;
import com.metawebthree.inventory.infrastructure.persistence.mapper.ProcessedEventMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.dao.DuplicateKeyException;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class ProcessedEventStoreTest {

    private ProcessedEventMapper mapper;
    private ProcessedEventStore store;

    @BeforeEach
    void setUp() {
        mapper = mock(ProcessedEventMapper.class);
        store = new ProcessedEventStore(mapper);
    }

    @Test
    void firstDeliveryIsProcessed() {
        when(mapper.countByEvent("evt-1", "inventory-service")).thenReturn(0);
        when(mapper.insert(any(ProcessedEvent.class))).thenReturn(1);

        assertTrue(store.markFirstProcessing("evt-1", "inventory-service"));
        verify(mapper).insert(any(ProcessedEvent.class));
    }

    @Test
    void redeliveryIsIgnored() {
        when(mapper.countByEvent("evt-1", "inventory-service")).thenReturn(1);

        assertFalse(store.markFirstProcessing("evt-1", "inventory-service"));
        verify(mapper, never()).insert(any());
    }

    @Test
    void insertRaceIsTreatedAsDuplicate() {
        when(mapper.countByEvent("evt-1", "inventory-service")).thenReturn(0);
        when(mapper.insert(any(ProcessedEvent.class))).thenThrow(new DuplicateKeyException("dup"));

        assertFalse(store.markFirstProcessing("evt-1", "inventory-service"));
    }

    @Test
    void missingEventIdIsProcessed() {
        assertTrue(store.markFirstProcessing(null, "inventory-service"));
        verifyNoInteractions(mapper);
    }
}
