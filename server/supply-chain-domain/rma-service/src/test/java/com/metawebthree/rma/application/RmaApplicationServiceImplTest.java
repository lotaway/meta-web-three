package com.metawebthree.rma.application;

import com.metawebthree.rma.application.dto.CreateRmaRequest;
import com.metawebthree.rma.application.dto.MakeDispositionRequest;
import com.metawebthree.rma.application.dto.RecordInspectionRequest;
import com.metawebthree.rma.application.dto.RmaOrderDTO;
import com.metawebthree.rma.application.event.RmaCompletedEvent;
import com.metawebthree.rma.application.event.RmaCreatedEvent;
import com.metawebthree.rma.application.event.RmaDispositionExecutedEvent;
import com.metawebthree.rma.application.event.RmaInspectionCompletedEvent;
import com.metawebthree.rma.domain.entity.*;
import com.metawebthree.rma.domain.repository.*;
import com.metawebthree.rma.domain.service.RmaDomainService;
import com.metawebthree.rma.infrastructure.event.RmaDomainEventPublisher;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Captor;
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
class RmaApplicationServiceImplTest {

    @Mock private RmaDomainService rmaDomainService;
    @Mock private RmaOrderRepository rmaOrderRepository;
    @Mock private RmaOrderItemRepository rmaOrderItemRepository;
    @Mock private RmaInspectionRepository rmaInspectionRepository;
    @Mock private RmaDispositionRepository rmaDispositionRepository;
    @Mock private ReturnShippingRepository returnShippingRepository;
    @Mock private RmaDomainEventPublisher eventPublisher;

    @Captor private ArgumentCaptor<RmaCreatedEvent> createdEventCaptor;
    @Captor private ArgumentCaptor<RmaInspectionCompletedEvent> inspectionEventCaptor;
    @Captor private ArgumentCaptor<RmaDispositionExecutedEvent> dispositionEventCaptor;

    private RmaApplicationServiceImpl service;

    @BeforeEach
    void setUp() {
        service = new RmaApplicationServiceImpl(
                rmaDomainService, rmaOrderRepository, rmaOrderItemRepository,
                rmaInspectionRepository, rmaDispositionRepository,
                returnShippingRepository, eventPublisher);
    }

    @Test
    void createRma_shouldCreateOrderAndPublishEvent() {
        CreateRmaRequest request = new CreateRmaRequest();
        request.setOrderNo("ORDER-001");
        request.setCustomerId(1L);
        request.setCustomerName("Test Customer");
        request.setContactPhone("13800138000");
        request.setReasonCode("DAMAGED");
        request.setReasonDescription("Item damaged in transit");
        request.setWarehouseId(1L);
        request.setReturnType("REFUND");
        request.setCreatedBy("admin");

        CreateRmaRequest.CreateRmaItem itemReq = new CreateRmaRequest.CreateRmaItem();
        itemReq.setSkuCode("SKU001");
        itemReq.setSkuName("Test SKU");
        itemReq.setExpectedQuantity(5);
        itemReq.setUnitPrice(BigDecimal.valueOf(100));
        request.setItems(List.of(itemReq));

        RmaOrder order = new RmaOrder();
        order.setId(1L);
        order.setRmaNo("RMA20260717-001");
        order.setOrderNo("ORDER-001");

        when(rmaDomainService.createRmaOrder(anyString(), anyLong(), anyString(), anyString(),
                anyString(), anyString(), anyLong(), anyString(), anyString(), anyList()))
                .thenReturn(order);

        RmaOrderDTO result = service.createRma(request);

        assertNotNull(result);
        verify(rmaDomainService).createRmaOrder(eq("ORDER-001"), eq(1L), eq("Test Customer"),
                eq("13800138000"), eq("DAMAGED"), eq("Item damaged in transit"),
                eq(1L), eq("REFUND"), eq("admin"), anyList());
        verify(rmaDomainService).saveRmaOrder(same(order), anyList());
        verify(eventPublisher).publish(createdEventCaptor.capture());
        assertEquals("RMA_CREATED", createdEventCaptor.getValue().getEventType());
        assertEquals(1L, createdEventCaptor.getValue().getRmaId());
    }

    @Test
    void getRma_whenExists_shouldReturnDTO() {
        RmaOrder order = new RmaOrder();
        order.setId(1L);
        order.setRmaNo("RMA001");
        order.setOrderNo("ORDER-001");
        order.setStatus(com.metawebthree.rma.domain.RmaOrderStatus.PENDING);
        order.setCreatedAt(LocalDateTime.now());
        order.setUpdatedAt(LocalDateTime.now());

        when(rmaOrderRepository.findById(1L)).thenReturn(Optional.of(order));
        when(rmaOrderItemRepository.findByRmaId(1L)).thenReturn(List.of());
        when(rmaInspectionRepository.findByRmaId(1L)).thenReturn(Optional.empty());
        when(rmaDispositionRepository.findByRmaId(1L)).thenReturn(Optional.empty());
        when(returnShippingRepository.findByRmaId(1L)).thenReturn(Optional.empty());

        RmaOrderDTO result = service.getRma(1L);

        assertNotNull(result);
        assertEquals("RMA001", result.getRmaNo());
    }

    @Test
    void getRma_whenNotFound_shouldThrow() {
        when(rmaOrderRepository.findById(99L)).thenReturn(Optional.empty());
        assertThrows(IllegalArgumentException.class, () -> service.getRma(99L));
    }

    @Test
    void recordInspection_shouldSaveAndPublishEvent() {
        Long rmaId = 1L;
        RecordInspectionRequest request = new RecordInspectionRequest();
        request.setInspector("Inspector A");
        request.setResult("PASS");
        request.setConclusion("NO_ISSUE");
        request.setTotalInspected(10);
        request.setTotalPassed(8);
        request.setTotalFailed(2);
        request.setRemark("All good");

        RmaOrder order = new RmaOrder();
        order.setId(rmaId);
        order.setRmaNo("RMA001");

        when(rmaDomainService.getRmaOrder(rmaId)).thenReturn(Optional.of(order));
        RmaInspection savedInspection = new RmaInspection();
        savedInspection.setId(1L);
        savedInspection.setResult("PASS");
        savedInspection.setTotalPassed(8);
        when(rmaDomainService.recordInspection(same(order), any())).thenReturn(savedInspection);

        service.recordInspection(rmaId, request);

        verify(rmaDomainService).recordInspection(same(order), any());
        verify(rmaDomainService).saveInspection(any());
        verify(rmaDomainService).saveRmaOrder(same(order));
        verify(eventPublisher).publish(inspectionEventCaptor.capture());
        assertEquals("RMA_INSPECTION_COMPLETED", inspectionEventCaptor.getValue().getEventType());
    }

    @Test
    void makeDisposition_shouldSetRmaIdAndSave() {
        Long rmaId = 1L;
        MakeDispositionRequest request = new MakeDispositionRequest();
        request.setDispositionType("REFUND");
        request.setDispositionBy("Admin");
        request.setRefundAmount(BigDecimal.valueOf(500));

        RmaOrder order = new RmaOrder();
        order.setId(rmaId);

        when(rmaDomainService.getRmaOrder(rmaId)).thenReturn(Optional.of(order));

        service.makeDisposition(rmaId, request);

        verify(rmaDomainService).makeDisposition(same(order), argThat(d ->
                d.getRmaId() != null && d.getRmaId().equals(rmaId)
        ));
    }

    @Test
    void executeDisposition_whenDispositionExists_shouldPublishEvent() {
        Long rmaId = 1L;
        RmaOrder order = new RmaOrder();
        order.setId(rmaId);
        order.setRmaNo("RMA001");

        RmaDisposition disposition = new RmaDisposition();
        disposition.setId(1L);
        disposition.setDispositionType("REFUND");

        when(rmaDomainService.executeDisposition(rmaId)).thenReturn(order);
        when(rmaDispositionRepository.findByRmaId(rmaId)).thenReturn(Optional.of(disposition));

        service.executeDisposition(rmaId);

        verify(eventPublisher).publish(dispositionEventCaptor.capture());
        assertEquals("RMA_DISPOSITION_EXECUTED", dispositionEventCaptor.getValue().getEventType());
    }

    @Test
    void completeRma_shouldCompleteAndPublishEvent() {
        Long rmaId = 1L;
        RmaOrder order = new RmaOrder();
        order.setId(rmaId);
        order.setRmaNo("RMA001");

        when(rmaDomainService.completeRmaOrder(rmaId)).thenReturn(order);

        service.completeRma(rmaId);

        verify(eventPublisher).publish(any(RmaCompletedEvent.class));
    }

    @Test
    void cancelRma_shouldCancelOrder() {
        Long rmaId = 1L;
        RmaOrder order = new RmaOrder();
        order.setId(rmaId);

        when(rmaDomainService.cancelRmaOrder(rmaId)).thenReturn(order);

        service.cancelRma(rmaId);

        verify(rmaDomainService).saveRmaOrder(same(order));
    }

    @Test
    void getRmaTimeline_shouldReturnOrderedEntities() {
        Long rmaId = 1L;
        RmaOrder order = new RmaOrder();
        RmaInspection inspection = new RmaInspection();
        RmaDisposition disposition = new RmaDisposition();
        ReturnShipping shipping = new ReturnShipping();

        when(rmaOrderRepository.findById(rmaId)).thenReturn(Optional.of(order));
        when(rmaInspectionRepository.findByRmaId(rmaId)).thenReturn(Optional.of(inspection));
        when(rmaDispositionRepository.findByRmaId(rmaId)).thenReturn(Optional.of(disposition));
        when(returnShippingRepository.findByRmaId(rmaId)).thenReturn(Optional.of(shipping));

        List<?> timeline = service.getRmaTimeline(rmaId);

        assertEquals(4, timeline.size());
    }

    @Test
    void toOrderDTO_withAllRelations_shouldMapCorrectly() {
        RmaOrder order = new RmaOrder();
        order.setId(1L);
        order.setRmaNo("RMA001");
        order.setOrderNo("ORDER-001");
        order.setStatus(com.metawebthree.rma.domain.RmaOrderStatus.PENDING);
        order.setReturnType("REFUND");
        order.setCustomerId(1L);
        order.setCustomerName("Test");
        order.setTotalQuantity(5);
        order.setTotalAmount(BigDecimal.valueOf(500));
        order.setCurrency("USD");
        order.setCreatedBy("admin");
        order.setCreatedAt(LocalDateTime.now());
        order.setUpdatedAt(LocalDateTime.now());

        RmaOrderItem item = new RmaOrderItem();
        item.setId(1L);
        item.setSkuCode("SKU001");
        item.setExpectedQuantity(5);
        item.setUnitPrice(BigDecimal.valueOf(100));

        when(rmaOrderRepository.findById(1L)).thenReturn(Optional.of(order));
        when(rmaOrderItemRepository.findByRmaId(1L)).thenReturn(List.of(item));
        when(rmaInspectionRepository.findByRmaId(1L)).thenReturn(Optional.empty());
        when(rmaDispositionRepository.findByRmaId(1L)).thenReturn(Optional.empty());
        when(returnShippingRepository.findByRmaId(1L)).thenReturn(Optional.empty());

        RmaOrderDTO result = service.getRma(1L);

        assertNotNull(result);
        assertEquals("RMA001", result.getRmaNo());
        assertNotNull(result.getItems());
        assertEquals(1, result.getItems().size());
        assertEquals("SKU001", result.getItems().get(0).getSkuCode());
    }
}
