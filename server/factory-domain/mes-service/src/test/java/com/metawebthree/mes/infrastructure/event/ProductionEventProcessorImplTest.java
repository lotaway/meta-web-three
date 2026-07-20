package com.metawebthree.mes.infrastructure.event;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.mes.domain.service.MesDomainService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.HashMap;
import java.util.Map;

import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class ProductionEventProcessorImplTest {

    @Mock
    private MesDomainService mesDomainService;
    @Mock
    private ObjectMapper objectMapper;

    private ProductionEventProcessorImpl processor;

    @BeforeEach
    void setUp() {
        processor = new ProductionEventProcessorImpl(mesDomainService, objectMapper);
    }

    @Test
    void handleOrderCreated_withValidData_shouldCreateWorkOrder() {
        Map<String, Object> eventData = new HashMap<>();
        eventData.put("orderCode", "ORD-001");
        eventData.put("productCode", "P001");

        processor.handleOrderCreated(eventData);

        verify(mesDomainService).createWorkOrder(
                "WO-ORD-001",
                "P001",
                "From ERP Order: ORD-001",
                0, null, null);
    }

    @Test
    void handleOrderCreated_whenServiceThrows_shouldNotRethrow() {
        Map<String, Object> eventData = new HashMap<>();
        eventData.put("orderCode", "ORD-001");
        eventData.put("productCode", "P001");
        doThrow(new RuntimeException("DB error"))
                .when(mesDomainService).createWorkOrder(anyString(), anyString(), anyString(), anyInt(), any(), any());

        processor.handleOrderCreated(eventData);

        verify(mesDomainService).createWorkOrder(anyString(), anyString(), anyString(), anyInt(), any(), any());
    }

    @Test
    void handleOrderCreated_withMissingFields_shouldStillCallService() {
        Map<String, Object> eventData = new HashMap<>();
        eventData.put("orderCode", "ORD-002");

        processor.handleOrderCreated(eventData);

        verify(mesDomainService).createWorkOrder(
                "WO-ORD-002",
                null,
                "From ERP Order: ORD-002",
                0, null, null);
    }
}
