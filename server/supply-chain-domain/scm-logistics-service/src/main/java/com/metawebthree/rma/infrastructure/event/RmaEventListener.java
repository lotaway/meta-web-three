package com.metawebthree.rma.infrastructure.event;

import com.metawebthree.rma.application.event.RmaCompletedEvent;
import com.metawebthree.rma.application.event.RmaDispositionExecutedEvent;
import com.metawebthree.rma.application.event.RmaInspectionCompletedEvent;
import com.metawebthree.rma.application.integration.InventoryIntegrationService;
import com.metawebthree.rma.application.integration.SettlementIntegrationService;
import com.metawebthree.rma.domain.entity.RmaDisposition;
import com.metawebthree.rma.domain.entity.RmaOrder;
import com.metawebthree.rma.domain.entity.RmaOrderItem;
import com.metawebthree.rma.domain.repository.RmaDispositionRepository;
import com.metawebthree.rma.domain.repository.RmaOrderItemRepository;
import com.metawebthree.rma.domain.repository.RmaOrderRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

/**
 * Listens to RMA domain events and triggers integration with inventory and settlement services.
 */
@Slf4j
@Component
public class RmaEventListener {

    private final InventoryIntegrationService inventoryIntegrationService;
    private final SettlementIntegrationService settlementIntegrationService;
    private final RmaOrderRepository rmaOrderRepository;
    private final RmaOrderItemRepository rmaOrderItemRepository;
    private final RmaDispositionRepository rmaDispositionRepository;

    public RmaEventListener(InventoryIntegrationService inventoryIntegrationService,
                            SettlementIntegrationService settlementIntegrationService,
                            RmaOrderRepository rmaOrderRepository,
                            RmaOrderItemRepository rmaOrderItemRepository,
                            RmaDispositionRepository rmaDispositionRepository) {
        this.inventoryIntegrationService = inventoryIntegrationService;
        this.settlementIntegrationService = settlementIntegrationService;
        this.rmaOrderRepository = rmaOrderRepository;
        this.rmaOrderItemRepository = rmaOrderItemRepository;
        this.rmaDispositionRepository = rmaDispositionRepository;
    }

    @EventListener
    @Transactional
    public void handleRmaInspectionCompleted(RmaInspectionCompletedEvent event) {
        log.info("Handling RmaInspectionCompletedEvent: rmaId={}, rmaNo={}, result={}, acceptedQuantity={}",
                event.getRmaId(), event.getRmaNo(), event.getInspectionResult(), event.getAcceptedQuantity());

        if (event.getAcceptedQuantity() == null || event.getAcceptedQuantity() <= 0) {
            log.info("No accepted items to stock in for rmaId={}", event.getRmaId());
            return;
        }

        RmaOrder order = rmaOrderRepository.findById(event.getRmaId()).orElse(null);
        if (order == null) {
            log.warn("RMA order not found for rmaId={}, cannot process inventory stock-in", event.getRmaId());
            return;
        }

        List<RmaOrderItem> items = rmaOrderItemRepository.findByRmaId(event.getRmaId());
        if (items == null || items.isEmpty()) {
            log.warn("No RMA items found for rmaId={}, cannot process inventory stock-in", event.getRmaId());
            return;
        }

        for (RmaOrderItem item : items) {
            Integer acceptedQty = item.getAcceptedQuantity() != null ? item.getAcceptedQuantity() : 0;
            if (acceptedQty <= 0) {
                continue;
            }
            inventoryIntegrationService.stockIn(
                    event.getRmaId(),
                    event.getRmaNo(),
                    item.getSkuCode(),
                    item.getSkuName(),
                    acceptedQty,
                    order.getWarehouseId()
            );
        }
    }

    @EventListener
    @Transactional
    public void handleRmaDispositionExecuted(RmaDispositionExecutedEvent event) {
        log.info("Handling RmaDispositionExecutedEvent: rmaId={}, rmaNo={}, dispositionType={}",
                event.getRmaId(), event.getRmaNo(), event.getDispositionType());

        RmaOrder order = rmaOrderRepository.findById(event.getRmaId()).orElse(null);
        if (order == null) {
            log.warn("RMA order not found for rmaId={}, cannot process settlement", event.getRmaId());
            return;
        }

        RmaDisposition disposition = rmaDispositionRepository.findByRmaId(event.getRmaId()).orElse(null);
        if (disposition == null) {
            log.warn("RMA disposition not found for rmaId={}, cannot process settlement", event.getRmaId());
            return;
        }

        if (disposition.getRefundAmount() != null
                && disposition.getRefundAmount().compareTo(java.math.BigDecimal.ZERO) > 0) {
            settlementIntegrationService.triggerRefund(
                    event.getRmaId(),
                    event.getRmaNo(),
                    order.getOrderNo(),
                    order.getCustomerId(),
                    disposition.getRefundAmount(),
                    event.getDispositionType()
            );
        } else {
            log.info("No refund amount for rmaId={}, dispositionType={} — skipping settlement",
                    event.getRmaId(), event.getDispositionType());
        }
    }
}
