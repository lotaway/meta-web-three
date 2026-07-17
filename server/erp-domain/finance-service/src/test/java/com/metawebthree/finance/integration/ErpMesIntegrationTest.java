package com.metawebthree.finance.integration;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.event.EventType;
import com.metawebthree.finance.application.command.cost.CostCommandService;
import com.metawebthree.finance.domain.entity.cost.ActualCost;
import com.metawebthree.finance.domain.repository.cost.ActualCostRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
public class ErpMesIntegrationTest {

    @Autowired
    private KafkaTemplate<String, String> kafkaTemplate;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private CostCommandService costCommandService;

    @Autowired
    private ActualCostRepository actualCostRepository;

    @DynamicPropertySource
    static void configureProperties(DynamicPropertyRegistry registry) {
        registry.add("spring.kafka.bootstrap-servers", () -> "localhost:9092");
    }

    @BeforeEach
    void setUp() {
        List<ActualCost> existing = actualCostRepository.findByProductCode("TEST-PROD-001");
        existing.forEach(cost -> {
        });
    }

    @Test
    void testMesWorkOrderCompletionTriggersCostAccounting() throws Exception {
        Map<String, Object> eventData = new HashMap<>();
        eventData.put("eventId", UUID.randomUUID().toString());
        eventData.put("eventType", "WORK_ORDER_COMPLETED");
        eventData.put("workOrderId", 10001L);
        eventData.put("workOrderNo", "WO-2026-001");
        eventData.put("productCode", "TEST-PROD-001");
        eventData.put("quantity", 500);

        String message = objectMapper.writeValueAsString(eventData);

        kafkaTemplate.send(EventType.MES_WORK_ORDER_COMPLETED_TOPIC, message).get();

        Thread.sleep(2000);

        List<ActualCost> costs = actualCostRepository.findByProductCode("TEST-PROD-001");
        assertFalse(costs.isEmpty(), "Expected at least one actual cost record to be created");

        ActualCost cost = costs.get(costs.size() - 1);
        assertEquals("WO-2026-001", cost.getProductionOrderNo());
        assertEquals("TEST-PROD-001", cost.getProductCode());
    }

    @Test
    void testMesTaskCompletedEventIsLogged() throws Exception {
        Map<String, Object> eventData = new HashMap<>();
        eventData.put("eventId", UUID.randomUUID().toString());
        eventData.put("eventType", "TASK_COMPLETED");
        eventData.put("taskId", 20001L);
        eventData.put("taskNo", "TK-2026-001");
        eventData.put("workOrderId", 10001L);
        eventData.put("workOrderNo", "WO-2026-001");
        eventData.put("productCode", "TEST-PROD-001");
        eventData.put("qualifiedQuantity", 480);
        eventData.put("defectiveQuantity", 20);

        String message = objectMapper.writeValueAsString(eventData);

        kafkaTemplate.send(EventType.MES_TASK_COMPLETED_TOPIC, message).get();

        Thread.sleep(1000);
    }

    @Test
    void testEndToEndFlow() throws Exception {
        String productionEvent = "{\"event\":\"ORDER_CREATED\",\"orderCode\":\"PO-2026-001\",\"productCode\":\"PROD-X\"}";
        kafkaTemplate.send(EventType.PRODUCTION_EVENTS_TOPIC, productionEvent).get();
        Thread.sleep(1000);

        Map<String, Object> completionEvent = new HashMap<>();
        completionEvent.put("eventId", UUID.randomUUID().toString());
        completionEvent.put("eventType", "WORK_ORDER_COMPLETED");
        completionEvent.put("workOrderId", 10002L);
        completionEvent.put("workOrderNo", "WO-PO-2026-001");
        completionEvent.put("productCode", "PROD-X");
        completionEvent.put("quantity", 200);

        String completionMessage = objectMapper.writeValueAsString(completionEvent);
        kafkaTemplate.send(EventType.MES_WORK_ORDER_COMPLETED_TOPIC, completionMessage).get();

        Thread.sleep(2000);

        List<ActualCost> costs = actualCostRepository.findByProductCode("PROD-X");
        assertFalse(costs.isEmpty(), "End-to-end flow should create actual cost records");

        ActualCost cost = costs.get(costs.size() - 1);
        assertEquals("WO-PO-2026-001", cost.getProductionOrderNo());
        assertEquals("PROD-X", cost.getProductCode());
        assertNotNull(cost.getQuantity());
        assertTrue(cost.getQuantity().compareTo(java.math.BigDecimal.ZERO) > 0,
            "Quantity should be greater than zero");
    }
}
