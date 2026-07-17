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
        publishWorkOrderCompletion("TEST-PROD-001", 500, "WO-2026-001", 10001L);
        Thread.sleep(2000);
        assertActualCostCreated("TEST-PROD-001", "WO-2026-001");
    }

    private void publishWorkOrderCompletion(String productCode, int quantity, String workOrderNo, Long workOrderId) throws Exception {
        Map<String, Object> eventData = new HashMap<>();
        eventData.put("eventId", UUID.randomUUID().toString());
        eventData.put("eventType", "WORK_ORDER_COMPLETED");
        eventData.put("workOrderId", workOrderId);
        eventData.put("workOrderNo", workOrderNo);
        eventData.put("productCode", productCode);
        eventData.put("quantity", quantity);
        String message = objectMapper.writeValueAsString(eventData);
        kafkaTemplate.send(EventType.MES_WORK_ORDER_COMPLETED_TOPIC, message).get();
    }

    private void assertActualCostCreated(String productCode, String expectedOrderNo) {
        List<ActualCost> costs = actualCostRepository.findByProductCode(productCode);
        assertFalse(costs.isEmpty(), "Expected at least one actual cost record to be created");
        ActualCost cost = costs.get(costs.size() - 1);
        assertEquals(expectedOrderNo, cost.getProductionOrderNo());
        assertEquals(productCode, cost.getProductCode());
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
        publishOrderCreatedEvent("PO-2026-001", "PROD-X");
        Thread.sleep(1000);
        publishWorkOrderCompletion("PROD-X", 200, "WO-PO-2026-001", 10002L);
        Thread.sleep(2000);
        assertActualCostCreated("PROD-X", "WO-PO-2026-001");
        assertCostQuantityPositive("PROD-X");
    }

    private void publishOrderCreatedEvent(String orderCode, String productCode) throws Exception {
        String productionEvent = "{\"event\":\"ORDER_CREATED\",\"orderCode\":\"" + orderCode + "\",\"productCode\":\"" + productCode + "\"}";
        kafkaTemplate.send(EventType.PRODUCTION_EVENTS_TOPIC, productionEvent).get();
    }

    private void assertCostQuantityPositive(String productCode) {
        List<ActualCost> costs = actualCostRepository.findByProductCode(productCode);
        ActualCost cost = costs.get(costs.size() - 1);
        assertNotNull(cost.getQuantity());
        assertTrue(cost.getQuantity().compareTo(java.math.BigDecimal.ZERO) > 0,
            "Quantity should be greater than zero");
    }
}
