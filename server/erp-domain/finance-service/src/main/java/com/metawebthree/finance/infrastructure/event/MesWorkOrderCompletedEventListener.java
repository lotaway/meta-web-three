package com.metawebthree.finance.infrastructure.event;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.event.EventType;
import com.metawebthree.finance.application.command.cost.CostCommandService;
import com.metawebthree.finance.application.command.cost.dto.ActualCostCreateCommand;
import com.metawebthree.finance.application.event.WorkOrderCompletionProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.Map;

@Component
public class MesWorkOrderCompletedEventListener implements WorkOrderCompletionProcessor {

    private static final Logger log = LoggerFactory.getLogger(MesWorkOrderCompletedEventListener.class);
    private static final String DEFAULT_PRODUCT_CODE = "UNKNOWN";
    private static final String DEFAULT_COST_TYPE = "PRODUCTION";
    private static final String DEFAULT_CURRENCY = "CNY";
    private static final Long DEFAULT_CREATED_BY = 0L;

    private final CostCommandService costCommandService;
    private final ObjectMapper objectMapper;

    public MesWorkOrderCompletedEventListener(CostCommandService costCommandService,
                                               ObjectMapper objectMapper) {
        this.costCommandService = costCommandService;
        this.objectMapper = objectMapper;
    }

    @Override
    @KafkaListener(topics = EventType.MES_WORK_ORDER_COMPLETED_TOPIC, groupId = "finance-service")
    public void onWorkOrderCompleted(String message) {
        try {
            Map<String, Object> eventData = objectMapper.readValue(message, Map.class);
            String workOrderNo = (String) eventData.get("workOrderNo");
            if (workOrderNo == null) {
                log.warn("workOrderNo is null in work order completed event");
                return;
            }
            String productCode = resolveString(eventData.get("productCode"), DEFAULT_PRODUCT_CODE);
            Integer quantity = resolveQuantity(eventData.get("quantity"));
            createActualCostRecord(workOrderNo, productCode, quantity);
        } catch (Exception e) {
            log.error("Failed to process MES work order completed event: {}", message, e);
        }
    }

    private void createActualCostRecord(String workOrderNo, String productCode, Integer quantity) {
        ActualCostCreateCommand command = buildCostCommand(workOrderNo, productCode, quantity);
        costCommandService.createActualCost(command);
        log.info("Auto-created actual cost record for MES work order: {}", workOrderNo);
    }

    private ActualCostCreateCommand buildCostCommand(String workOrderNo, String productCode, Integer quantity) {
        ActualCostCreateCommand command = new ActualCostCreateCommand();
        command.setProductionOrderNo(workOrderNo);
        command.setProductCode(productCode);
        command.setProductName("From MES: " + workOrderNo);
        command.setCostDate(LocalDate.now());
        command.setQuantity(BigDecimal.valueOf(quantity));
        command.setActualMaterialCost(BigDecimal.ZERO);
        command.setActualLaborCost(BigDecimal.ZERO);
        command.setActualOverheadCost(BigDecimal.ZERO);
        command.setCostType(DEFAULT_COST_TYPE);
        command.setCreatedBy(DEFAULT_CREATED_BY);
        command.setCurrency(DEFAULT_CURRENCY);
        command.setRemark("Auto-created from MES work order completion: " + workOrderNo);
        return command;
    }

    @Override
    @KafkaListener(topics = EventType.MES_TASK_COMPLETED_TOPIC, groupId = "finance-service")
    public void onTaskCompleted(String message) {
        log.debug("Received MES task completed event: {}", message);
    }

    private String resolveString(Object value, String defaultValue) {
        return value != null ? (String) value : defaultValue;
    }

    private Integer resolveQuantity(Object value) {
        return value != null ? ((Number) value).intValue() : 0;
    }
}
