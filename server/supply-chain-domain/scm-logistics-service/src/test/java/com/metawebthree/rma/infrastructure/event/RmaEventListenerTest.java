package com.metawebthree.rma.infrastructure.event;

import com.metawebthree.rma.application.RmaStockInService;
import com.metawebthree.rma.application.event.RmaInspectionCompletedEvent;
import com.metawebthree.rma.application.integration.SettlementIntegrationService;
import com.metawebthree.rma.domain.entity.RmaOrder;
import com.metawebthree.rma.domain.entity.RmaOrderItem;
import com.metawebthree.rma.domain.repository.RmaDispositionRepository;
import com.metawebthree.rma.domain.repository.RmaOrderItemRepository;
import com.metawebthree.rma.domain.repository.RmaOrderRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class RmaEventListenerTest {

    private RmaStockInService rmaStockInService;
    private RmaOrderRepository rmaOrderRepository;
    private RmaOrderItemRepository rmaOrderItemRepository;
    private RmaEventListener listener;

    @BeforeEach
    void setUp() {
        rmaStockInService = mock(RmaStockInService.class);
        rmaOrderRepository = mock(RmaOrderRepository.class);
        rmaOrderItemRepository = mock(RmaOrderItemRepository.class);
        listener = new RmaEventListener(
                rmaStockInService,
                mock(SettlementIntegrationService.class),
                rmaOrderRepository,
                rmaOrderItemRepository,
                mock(RmaDispositionRepository.class));
    }

    private RmaOrder order(Long warehouseId) {
        RmaOrder order = new RmaOrder();
        order.setId(1L);
        order.setRmaNo("RMA001");
        order.setWarehouseId(warehouseId);
        return order;
    }

    private RmaOrderItem item(int acceptedQuantity) {
        RmaOrderItem item = new RmaOrderItem();
        item.setRmaId(1L);
        item.setSkuCode("SKU001");
        item.setSkuName("Widget");
        item.setAcceptedQuantity(acceptedQuantity);
        return item;
    }

    private RmaInspectionCompletedEvent event(String result, int accepted) {
        return new RmaInspectionCompletedEvent(
                "RMA_INSPECTION_COMPLETED", 1L, "RMA001", result, accepted);
    }

    @Test
    void acceptedItemIsDispatchedToStockIn() {
        when(rmaOrderRepository.findById(1L)).thenReturn(Optional.of(order(100L)));
        when(rmaOrderItemRepository.findByRmaId(1L)).thenReturn(List.of(item(3)));

        listener.handleRmaInspectionCompleted(event("PASS", 3));

        verify(rmaStockInService).dispatch(1L, "RMA001", "SKU001", "Widget", 3, 100L);
    }

    @Test
    void failedInspectionIsNotStockedIn() {
        listener.handleRmaInspectionCompleted(event("FAIL", 0));

        verify(rmaStockInService, never()).dispatch(any(), any(), any(), any(), anyInt(), any());
    }

    @Test
    void rejectedItemIsNotStockedIn() {
        when(rmaOrderRepository.findById(1L)).thenReturn(Optional.of(order(100L)));
        when(rmaOrderItemRepository.findByRmaId(1L)).thenReturn(List.of(item(0)));

        listener.handleRmaInspectionCompleted(event("PARTIAL", 0));

        verify(rmaStockInService, never()).dispatch(any(), any(), any(), any(), anyInt(), any());
    }

    @Test
    void missingRmaOrderIsNotStockedIn() {
        when(rmaOrderRepository.findById(1L)).thenReturn(Optional.empty());

        listener.handleRmaInspectionCompleted(event("PASS", 3));

        verify(rmaStockInService, never()).dispatch(any(), any(), any(), any(), anyInt(), any());
    }
}
