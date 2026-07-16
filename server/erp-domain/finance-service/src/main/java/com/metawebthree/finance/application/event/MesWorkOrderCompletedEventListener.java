package com.metawebthree.finance.application.event;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.finance.application.command.cost.CostCommandService;
import com.metawebthree.finance.application.command.cost.dto.ActualCostCreateCommand;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.Map;

/**
 * MES work order completion event listener.
 * Listens for MES work order completion events and automatically
 * creates actual cost records in the finance/cost accounting module.
 *
 * ERP-MES data closed-loop: MES报工完成 -> 财务成本自动归集
 */
@Component
public class MesWorkOrderCompletedEventListener {

    private static final Logger log = LoggerFactory.getLogger(MesWorkOrderCompletedEventListener.class);

    private final CostCommandService costCommandService;
    private final ObjectMapper objectMapper;

    public MesWorkOrderCompletedEventListener(CostCommandService costCommandService,
                                               ObjectMapper objectMapper) {
        this.costCommandService = costCommandService;
        this.objectMapper = objectMapper;
    }

    /**
     * Listens for MES work order completion events and creates actual cost records.
     * Topic: mes.work_order_completed (published by mes-service)
     */
    @KafkaListener(topics = "mes.work_order_completed", groupId = "finance-service")
    public void onWorkOrderCompleted(String message) {
        try {
            log.info("Received MES work order completed event: {}", message);
            Map<String, Object> eventData = objectMapper.readValue(message, Map.class);

            String workOrderNo = (String) eventData.get("workOrderNo");
            if (workOrderNo == null) {
                log.warn("workOrderNo is null in work order completed event");
                return;
            }

            String productCode = (String) eventData.get("productCode");
            Integer quantity = eventData.get("quantity") != null
                ? ((Number) eventData.get("quantity")).intValue()
                : 0;

            // Auto-create actual cost record from MES completion data
            ActualCostCreateCommand command = new ActualCostCreateCommand();
            command.setProductionOrderNo(workOrderNo);
            command.setProductCode(productCode != null ? productCode : "UNKNOWN");
            command.setProductName("From MES: " + workOrderNo);
            command.setCostDate(LocalDate.now());
            command.setQuantity(BigDecimal.valueOf(quantity));
            command.setActualMaterialCost(BigDecimal.ZERO);
            command.setActualLaborCost(BigDecimal.ZERO);
            command.setActualOverheadCost(BigDecimal.ZERO);
            command.setCostType("PRODUCTION");
            command.setCreatedBy(0L);
            command.setCurrency("CNY");
            command.setRemark("Auto-created from MES work order completion: " + workOrderNo);

            costCommandService.createActualCost(command);

            log.info("Auto-created actual cost record for MES work order: {}", workOrderNo);

        } catch (Exception e) {
            log.error("Failed to process MES work order completed event: {}", message, e);
        }
    }

    /**
     * Listens for MES task completed events for finer-grained cost data.
     * Topic: mes.task_completed (published by mes-service)
     */
    @KafkaListener(topics = "mes.task_completed", groupId = "finance-service")
    public void onTaskCompleted(String message) {
        try {
            log.debug("Received MES task completed event: {}", message);
            // Task-level completion can trigger more detailed cost allocation
            // For now, log the event; cost is captured at work order level
        } catch (Exception e) {
            log.error("Failed to process MES task completed event: {}", message, e);
        }
    }
}
