package com.metawebthree.rma.domain.service;

import com.metawebthree.rma.domain.RmaOrderStatus;
import com.metawebthree.rma.domain.RmaSequenceGenerator;
import com.metawebthree.rma.domain.entity.*;
import com.metawebthree.rma.domain.repository.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class RmaDomainServiceImplTest {

    @Mock private RmaOrderRepository rmaOrderRepository;
    @Mock private RmaOrderItemRepository rmaOrderItemRepository;
    @Mock private RmaInspectionRepository rmaInspectionRepository;
    @Mock private RmaDispositionRepository rmaDispositionRepository;
    @Mock private ReturnShippingRepository returnShippingRepository;
    @Mock private RmaSequenceGenerator rmaSequenceGenerator;

    private RmaDomainServiceImpl service;

    @BeforeEach
    void setUp() {
        service = new RmaDomainServiceImpl(rmaOrderRepository, rmaOrderItemRepository,
                rmaInspectionRepository, rmaDispositionRepository,
                returnShippingRepository, rmaSequenceGenerator);
    }

    @Test
    void createRmaOrder_shouldBuildOrderWithDefaults() {
        when(rmaSequenceGenerator.generateRmaNo()).thenReturn("RMA20260717-001");

        RmaOrderItem item = new RmaOrderItem();
        item.setSkuCode("SKU001");
        item.setExpectedQuantity(5);
        item.setUnitPrice(BigDecimal.valueOf(100));

        RmaOrder result = service.createRmaOrder("ORDER-001", 1L, "Test", "13800138000",
                "DAMAGED", "Damaged in transit", 1L, "REFUND", "admin", List.of(item));

        assertNotNull(result);
        assertEquals("RMA20260717-001", result.getRmaNo());
        assertEquals(RmaOrderStatus.PENDING, result.getStatus());
        assertEquals(5, result.getTotalQuantity());
        assertEquals(BigDecimal.valueOf(500), result.getTotalAmount());
        assertEquals("CNY", result.getCurrency());
        assertEquals(0, result.getVersion());
    }

    @Test
    void submitForInspection_whenPending_shouldSetAwaitingInspection() {
        RmaOrder order = new RmaOrder();
        order.setId(1L);
        order.setStatus(RmaOrderStatus.PENDING);
        when(rmaOrderRepository.findById(1L)).thenReturn(Optional.of(order));

        RmaOrder result = service.submitForInspection(1L);

        assertEquals(RmaOrderStatus.AWAITING_INSPECTION, result.getStatus());
    }

    @Test
    void submitForInspection_whenWrongStatus_shouldThrow() {
        RmaOrder order = new RmaOrder();
        order.setId(1L);
        order.setStatus(RmaOrderStatus.COMPLETED);
        when(rmaOrderRepository.findById(1L)).thenReturn(Optional.of(order));

        assertThrows(IllegalStateException.class, () -> service.submitForInspection(1L));
    }

    @Test
    void recordInspection_shouldSetInspectedAndUpdateItemQuantitiesProportionally() {
        RmaOrder order = new RmaOrder();
        order.setId(1L);
        order.setStatus(RmaOrderStatus.AWAITING_INSPECTION);

        RmaInspection inspection = new RmaInspection();
        inspection.setInspector("Inspector A");
        inspection.setResult("PASS");
        inspection.setConclusion("NO_ISSUE");
        inspection.setTotalInspected(10);
        inspection.setTotalPassed(8);
        inspection.setTotalFailed(2);

        RmaOrderItem item1 = new RmaOrderItem();
        item1.setId(1L);
        item1.setExpectedQuantity(10);
        RmaOrderItem item2 = new RmaOrderItem();
        item2.setId(2L);
        item2.setExpectedQuantity(20);

        when(rmaOrderItemRepository.findByRmaId(1L)).thenReturn(List.of(item1, item2));

        RmaInspection result = service.recordInspection(order, inspection);

        assertEquals(RmaOrderStatus.INSPECTED, order.getStatus());
        assertEquals("PASS", result.getResult());
        verify(rmaOrderItemRepository, times(2)).save(any());

        assertEquals(3, item1.getInspectedQuantity(), 1);
        assertEquals(3, item1.getAcceptedQuantity(), 1);
        assertEquals(7, item2.getInspectedQuantity(), 1);
        assertEquals(5, item2.getAcceptedQuantity(), 1);
    }

    @Test
    void recordInspection_whenNotAwaitingInspection_shouldThrow() {
        RmaOrder order = new RmaOrder();
        order.setStatus(RmaOrderStatus.COMPLETED);

        assertThrows(IllegalStateException.class,
                () -> service.recordInspection(order, new RmaInspection()));
    }

    @Test
    void makeDisposition_shouldSetRmaIdAndReturnDisposition() {
        RmaOrder order = new RmaOrder();
        order.setId(1L);
        order.setStatus(RmaOrderStatus.INSPECTED);

        RmaDisposition disposition = new RmaDisposition();
        disposition.setDispositionType("REFUND");

        RmaDisposition result = service.makeDisposition(order, disposition);

        assertEquals(1L, result.getRmaId());
        assertEquals("REFUND", result.getDispositionType());
        assertEquals(RmaOrderStatus.AWAITING_DISPOSITION, order.getStatus());
    }

    @Test
    void makeDisposition_whenCannotDispose_shouldThrow() {
        RmaOrder order = new RmaOrder();
        order.setStatus(RmaOrderStatus.PENDING);

        assertThrows(IllegalStateException.class,
                () -> service.makeDisposition(order, new RmaDisposition()));
    }

    @Test
    void executeDisposition_shouldSetDisposed() {
        RmaOrder order = new RmaOrder();
        order.setId(1L);
        order.setStatus(RmaOrderStatus.AWAITING_DISPOSITION);
        when(rmaOrderRepository.findById(1L)).thenReturn(Optional.of(order));

        RmaOrder result = service.executeDisposition(1L);

        assertEquals(RmaOrderStatus.DISPOSED, result.getStatus());
    }

    @Test
    void completeRmaOrder_whenDisposed_shouldComplete() {
        RmaOrder order = new RmaOrder();
        order.setId(1L);
        order.setStatus(RmaOrderStatus.DISPOSED);
        when(rmaOrderRepository.findById(1L)).thenReturn(Optional.of(order));

        RmaOrder result = service.completeRmaOrder(1L);

        assertEquals(RmaOrderStatus.COMPLETED, result.getStatus());
    }

    @Test
    void cancelRmaOrder_whenPending_shouldCancel() {
        RmaOrder order = new RmaOrder();
        order.setId(1L);
        order.setStatus(RmaOrderStatus.PENDING);
        when(rmaOrderRepository.findById(1L)).thenReturn(Optional.of(order));

        RmaOrder result = service.cancelRmaOrder(1L);

        assertEquals(RmaOrderStatus.CANCELLED, result.getStatus());
    }

    @Test
    void saveReturnShipping_shouldDelegateAndReturn() {
        ReturnShipping shipping = new ReturnShipping();
        shipping.setRmaId(1L);
        shipping.setTrackingNo("SF123456");
        when(returnShippingRepository.save(shipping)).thenReturn(shipping);

        ReturnShipping result = service.saveReturnShipping(shipping);

        assertNotNull(result);
        verify(returnShippingRepository).save(shipping);
    }

    @Test
    void getRmaOrder_whenExists_shouldReturn() {
        RmaOrder order = new RmaOrder();
        order.setId(1L);
        when(rmaOrderRepository.findById(1L)).thenReturn(Optional.of(order));

        Optional<RmaOrder> result = service.getRmaOrder(1L);

        assertTrue(result.isPresent());
        assertEquals(1L, result.get().getId());
    }

    @Test
    void getRmaOrder_whenNotExists_shouldReturnEmpty() {
        when(rmaOrderRepository.findById(99L)).thenReturn(Optional.empty());

        Optional<RmaOrder> result = service.getRmaOrder(99L);

        assertTrue(result.isEmpty());
    }
}
