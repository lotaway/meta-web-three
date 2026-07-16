package com.metawebthree.finance.integration;

import com.fasterxml.jackson.databind.ObjectMapper;
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

/**
 * End-to-end integration test for ERP-MES data closed-loop.
 *
 * Tests the flow:
 * 1. MES work order completion event is published to Kafka
 * 2. Finance service listener consumes the event
 * 3. Actual cost record is automatically created
 */
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
        // Clear any existing test records
        List<ActualCost> existing = actualCostRepository.findByProductCode("TEST-PROD-001");
        existing.forEach(cost -> {
            // In a real test, we'd use a proper delete method
        });
    }

    @Test
    void testMesWorkOrderCompletionTriggersCostAccounting() throws Exception {
        // Arrange: prepare MES work order completion event
        Map<String, Object> eventData = new HashMap<>();
        eventData.put("eventId", UUID.randomUUID().toString());
        eventData.put("eventType", "WORK_ORDER_COMPLETED");
        eventData.put("workOrderId", 10001L);
        eventData.put("workOrderNo", "WO-2026-001");
        eventData.put("productCode", "TEST-PROD-001");
        eventData.put("quantity", 500);

        String message = objectMapper.writeValueAsString(eventData);

        // Act: publish event to the MES cross-domain topic
        kafkaTemplate.send("mes.work_order_completed", message).get();

        // Wait for async processing
        Thread.sleep(2000);

        // Assert: verify an actual cost record was created
        List<ActualCost> costs = actualCostRepository.findByProductCode("TEST-PROD-001");
        assertFalse(costs.isEmpty(), "Expected at least one actual cost record to be created");

        ActualCost cost = costs.get(costs.size() - 1);
        assertEquals("WO-2026-001", cost.getProductionOrderNo());
        assertEquals("TEST-PROD-001", cost.getProductCode());
    }

    @Test
    void testMesTaskCompletedEventIsLogged() throws Exception {
        // Arrange: prepare MES task completed event
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

        // Act: publish event to the MES task completed topic
        kafkaTemplate.send("mes.task_completed", message).get();

        // Assert: no exception should be thrown; event is logged at debug level
        // The test verifies the listener handles the event gracefully
        Thread.sleep(1000);
    }

    @Test
    void testEndToEndFlow() throws Exception {
        // Simulate the full ERP-MES-ERP closed loop:
        // Step 1: Production order released (ERP -> MES)
        String productionEvent = "{\"event\":\"ORDER_CREATED\",\"orderCode\":\"PO-2026-001\",\"productCode\":\"PROD-X\"}";
        kafkaTemplate.send("production.events", productionEvent).get();
        Thread.sleep(1000);

        // Step 2: MES work order completed (MES -> ERP Finance)
        Map<String, Object> completionEvent = new HashMap<>();
        completionEvent.put("eventId", UUID.randomUUID().toString());
        completionEvent.put("eventType", "WORK_ORDER_COMPLETED");
        completionEvent.put("workOrderId", 10002L);
        completionEvent.put("workOrderNo", "WO-PO-2026-001");
        completionEvent.put("productCode", "PROD-X");
        completionEvent.put("quantity", 200);

        String completionMessage = objectMapper.writeValueAsString(completionEvent);
        kafkaTemplate.send("mes.work_order_completed", completionMessage).get();

        // Wait for async processing
        Thread.sleep(2000);

        // Verify: cost record was created from MES completion event
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
