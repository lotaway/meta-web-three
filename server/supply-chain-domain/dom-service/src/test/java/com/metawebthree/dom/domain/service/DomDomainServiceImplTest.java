package com.metawebthree.dom.domain.service;

import com.metawebthree.dom.domain.entity.*;
import com.metawebthree.dom.domain.repository.DomOrderLineRepository;
import com.metawebthree.dom.domain.repository.DomOrderRepository;
import com.metawebthree.dom.domain.repository.FulfillmentPlanRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class DomDomainServiceImplTest {

    @Mock private DomOrderRepository domOrderRepository;
    @Mock private DomOrderLineRepository domOrderLineRepository;
    @Mock private FulfillmentPlanRepository fulfillmentPlanRepository;
    @Mock private InventoryServiceClient inventoryServiceClient;
    @Mock private WarehouseServiceClient warehouseServiceClient;
    @Mock private DomDomainEventPublisher eventPublisher;
    @Mock private DomSequenceGenerator sequenceGenerator;

    private DomSourcingProperties sourcingProperties;
    private DomDomainServiceImpl domainService;

    @BeforeEach
    void setUp() {
        sourcingProperties = new DomSourcingProperties();
        sourcingProperties.setWarehouseIds(Arrays.asList(1L, 2L));
        sourcingProperties.setShippingCostPerKm(0.5);
        sourcingProperties.setHandlingCostFlat(10.0);
        sourcingProperties.setDistanceScoreFactor(100.0);
        sourcingProperties.setCostScoreFactor(1000.0);
        sourcingProperties.setBalancedDistanceWeight(0.5);
        sourcingProperties.setBalancedCostWeight(0.5);
        sourcingProperties.setWhNamePrefix("WH-");

        domainService = new DomDomainServiceImpl(
                domOrderRepository, domOrderLineRepository,
                fulfillmentPlanRepository, inventoryServiceClient,
                warehouseServiceClient, eventPublisher,
                sequenceGenerator, sourcingProperties);
    }

    @Test
    void createDomOrder_shouldGenerateOrderNoAndSetDefaults() {
        when(sequenceGenerator.generateDomOrderNo()).thenReturn("DOM20260717-000001");

        DomOrder order = new DomOrder();
        order.setOriginalOrderNo("ORDER-001");
        List<DomOrderLine> lines = List.of();

        DomOrder result = domainService.createDomOrder(order, lines);

        assertEquals("DOM20260717-000001", result.getDomOrderNo());
        assertEquals(DomOrderStatus.PENDING, result.getStatus());
        assertNotNull(result.getCreatedAt());
        assertNotNull(result.getUpdatedAt());
        assertEquals(0, result.getVersion());
    }

    @Test
    void saveDomOrder_shouldSaveOrderAndLinesAndPublishEvent() {
        DomOrder order = new DomOrder();
        order.setId(1L);
        DomOrderLine line = new DomOrderLine();
        line.setQuantity(5);

        when(domOrderRepository.save(same(order))).thenReturn(order);

        domainService.saveDomOrder(order, List.of(line));

        verify(domOrderRepository).save(same(order));
        verify(domOrderLineRepository).save(same(line));
        assertEquals(DomOrderLineStatus.PENDING, line.getStatus());
        assertEquals(0, line.getFulfilledQuantity());
        assertNotNull(line.getCreatedAt());
        verify(eventPublisher).publishDomOrderCreated(same(order));
    }

    @Test
    void checkAvailability_whenAllSufficient_shouldReturnTrue() {
        when(inventoryServiceClient.checkInventory("SKU001", 1L)).thenReturn(100);
        when(inventoryServiceClient.checkInventory("SKU001", 2L)).thenReturn(50);

        DomOrder order = new DomOrder();
        DomOrderLine line = new DomOrderLine();
        line.setSkuCode("SKU001");
        line.setQuantity(30);
        List<DomOrderLine> lines = List.of(line);

        boolean result = domainService.checkAvailability(order, lines);

        assertTrue(result);
        assertEquals(DomOrderLineStatus.ATP_PASS, line.getStatus());
        verify(inventoryServiceClient, times(2)).checkInventory(eq("SKU001"), anyLong());
    }

    @Test
    void checkAvailability_whenInsufficient_shouldReturnFalse() {
        when(inventoryServiceClient.checkInventory("SKU001", 1L)).thenReturn(10);
        when(inventoryServiceClient.checkInventory("SKU001", 2L)).thenReturn(5);

        DomOrder order = new DomOrder();
        DomOrderLine line = new DomOrderLine();
        line.setSkuCode("SKU001");
        line.setQuantity(30);
        List<DomOrderLine> lines = List.of(line);

        boolean result = domainService.checkAvailability(order, lines);

        assertFalse(result);
        assertEquals(DomOrderLineStatus.ATP_FAIL, line.getStatus());
    }

    @Test
    void checkAvailability_shouldSumAcrossAllWarehouses() {
        when(inventoryServiceClient.checkInventory("SKU001", 1L)).thenReturn(100);
        when(inventoryServiceClient.checkInventory("SKU001", 2L)).thenReturn(50);

        DomOrder order = new DomOrder();
        DomOrderLine line = new DomOrderLine();
        line.setSkuCode("SKU001");
        line.setQuantity(150);
        List<DomOrderLine> lines = List.of(line);

        boolean result = domainService.checkAvailability(order, lines);

        assertTrue(result);
    }

    @Test
    void checkAvailability_withNullInventory_shouldTreatAsZero() {
        when(inventoryServiceClient.checkInventory("SKU001", 1L)).thenReturn(null);
        when(inventoryServiceClient.checkInventory("SKU001", 2L)).thenReturn(50);

        DomOrder order = new DomOrder();
        DomOrderLine line = new DomOrderLine();
        line.setSkuCode("SKU001");
        line.setQuantity(60);
        List<DomOrderLine> lines = List.of(line);

        boolean result = domainService.checkAvailability(order, lines);

        assertFalse(result);
    }

    @Test
    void saveAvailabilityResult_whenAllPass_shouldSetSourcingStatus() {
        DomOrder order = new DomOrder();
        DomOrderLine line = new DomOrderLine();
        line.setStatus(DomOrderLineStatus.ATP_PASS);
        List<DomOrderLine> lines = List.of(line);

        domainService.saveAvailabilityResult(order, lines, true);

        assertEquals(DomOrderStatus.SOURCING, order.getStatus());
        verify(domOrderLineRepository).save(same(line));
        verify(domOrderRepository).save(same(order));
    }

    @Test
    void saveAvailabilityResult_whenAnyFail_shouldSetAtpFailedStatus() {
        DomOrder order = new DomOrder();
        DomOrderLine line = new DomOrderLine();
        line.setStatus(DomOrderLineStatus.ATP_FAIL);
        List<DomOrderLine> lines = List.of(line);

        domainService.saveAvailabilityResult(order, lines, false);

        assertEquals(DomOrderStatus.ATP_FAILED, order.getStatus());
    }

    @Test
    void sourceOrder_withBalancedStrategy_shouldScoreAndSelectBest() {
        DomOrderLine line = new DomOrderLine();
        line.setId(1L);
        line.setSkuCode("SKU001");
        line.setQuantity(10);
        line.setStatus(DomOrderLineStatus.ATP_PASS);

        DomOrder order = new DomOrder();
        order.setRegion("华东");

        when(inventoryServiceClient.checkInventory("SKU001", 1L)).thenReturn(100);
        when(inventoryServiceClient.checkInventory("SKU001", 2L)).thenReturn(50);

        WarehouseInfo wh1 = new WarehouseInfo(1L, "WH-East", "华东", 31.23, 121.47);
        WarehouseInfo wh2 = new WarehouseInfo(2L, "WH-North", "华北", 39.90, 116.40);

        when(warehouseServiceClient.getWarehouse(1L)).thenReturn(wh1);
        when(warehouseServiceClient.getWarehouse(2L)).thenReturn(wh2);
        when(warehouseServiceClient.getWarehouseDistance("华东", 1L)).thenReturn(50.0);
        when(warehouseServiceClient.getWarehouseDistance("华东", 2L)).thenReturn(500.0);

        List<DomOrderLine> result = domainService.sourceOrder(order, List.of(line), SourcingStrategy.BALANCED);

        assertEquals(1, result.size());
        assertEquals(DomOrderLineStatus.SOURCED, result.get(0).getStatus());
        assertEquals(1L, result.get(0).getWarehouseId());
        assertNotNull(result.get(0).getWarehouseName());
    }

    @Test
    void sourceOrder_withNearestStrategy_shouldPickNearest() {
        DomOrderLine line = new DomOrderLine();
        line.setId(1L);
        line.setSkuCode("SKU001");
        line.setQuantity(10);
        line.setStatus(DomOrderLineStatus.ATP_PASS);

        DomOrder order = new DomOrder();
        order.setRegion("华北");

        when(inventoryServiceClient.checkInventory("SKU001", 1L)).thenReturn(100);
        when(inventoryServiceClient.checkInventory("SKU001", 2L)).thenReturn(50);

        WarehouseInfo wh1 = new WarehouseInfo(1L, "WH-East", "华东", 31.23, 121.47);
        WarehouseInfo wh2 = new WarehouseInfo(2L, "WH-North", "华北", 39.90, 116.40);

        when(warehouseServiceClient.getWarehouse(1L)).thenReturn(wh1);
        when(warehouseServiceClient.getWarehouse(2L)).thenReturn(wh2);
        when(warehouseServiceClient.getWarehouseDistance("华北", 1L)).thenReturn(800.0);
        when(warehouseServiceClient.getWarehouseDistance("华北", 2L)).thenReturn(50.0);

        List<DomOrderLine> result = domainService.sourceOrder(order, List.of(line), SourcingStrategy.NEAREST_WAREHOUSE);

        assertEquals(2L, result.get(0).getWarehouseId());
    }

    @Test
    void sourceOrder_whenNoWarehouseAvailable_shouldMarkFailed() {
        DomOrderLine line = new DomOrderLine();
        line.setId(1L);
        line.setSkuCode("SKU001");
        line.setQuantity(10);
        line.setStatus(DomOrderLineStatus.ATP_PASS);

        DomOrder order = new DomOrder();
        order.setRegion("华东");

        when(inventoryServiceClient.checkInventory("SKU001", 1L)).thenReturn(0);
        when(inventoryServiceClient.checkInventory("SKU001", 2L)).thenReturn(0);

        List<DomOrderLine> result = domainService.sourceOrder(order, List.of(line), SourcingStrategy.BALANCED);

        assertEquals(1, result.size());
        assertEquals(DomOrderLineStatus.SOURCING_FAILED, result.get(0).getStatus());
    }

    @Test
    void sourceOrder_shouldSkipNonAtpPassLines() {
        DomOrderLine passLine = new DomOrderLine();
        passLine.setId(1L);
        passLine.setSkuCode("SKU001");
        passLine.setQuantity(10);
        passLine.setStatus(DomOrderLineStatus.ATP_PASS);

        DomOrderLine failLine = new DomOrderLine();
        failLine.setId(2L);
        failLine.setSkuCode("SKU002");
        failLine.setQuantity(10);
        failLine.setStatus(DomOrderLineStatus.ATP_FAIL);

        DomOrder order = new DomOrder();

        when(inventoryServiceClient.checkInventory("SKU001", 1L)).thenReturn(100);
        when(inventoryServiceClient.checkInventory("SKU001", 2L)).thenReturn(50);
        WarehouseInfo info = new WarehouseInfo(1L, "WH", "华东", 0, 0);
        when(warehouseServiceClient.getWarehouse(anyLong())).thenReturn(info);
        when(warehouseServiceClient.getWarehouseDistance(any(), anyLong())).thenReturn(100.0);

        List<DomOrderLine> result = domainService.sourceOrder(order, List.of(passLine, failLine), SourcingStrategy.LOWEST_COST);

        assertEquals(1, result.size());
        assertEquals(DomOrderLineStatus.SOURCED, result.get(0).getStatus());
    }

    @Test
    void saveSourcingResult_whenAllSourced_shouldComplete() {
        DomOrder order = new DomOrder();
        DomOrderLine line = new DomOrderLine();
        line.setStatus(DomOrderLineStatus.SOURCED);
        List<DomOrderLine> lines = List.of(line);

        domainService.saveSourcingResult(order, lines);

        assertEquals(DomOrderStatus.SOURCING_COMPLETED, order.getStatus());
        verify(domOrderRepository).save(same(order));
    }

    @Test
    void createFulfillmentPlan_shouldCalculateCorrectCounts() {
        DomOrder order = new DomOrder();
        order.setId(1L);
        order.setDomOrderNo("DOM001");

        DomOrderLine sourced = new DomOrderLine();
        sourced.setStatus(DomOrderLineStatus.SOURCED);
        DomOrderLine failed = new DomOrderLine();
        failed.setStatus(DomOrderLineStatus.SOURCING_FAILED);

        FulfillmentPlan plan = domainService.createFulfillmentPlan(order, List.of(sourced, failed));

        assertEquals(1L, plan.getDomOrderId());
        assertEquals(2, plan.getTotalLines());
        assertEquals(1, plan.getFulfilledLines());
        assertEquals(1, plan.getUnfulfilledLines());
        assertEquals(0, plan.getPartiallyFulfilledLines());
        assertEquals(FulfillmentPlanStatus.DRAFT, plan.getStatus());
        assertNotNull(plan.getCreatedAt());
    }

    @Test
    void approveFulfillmentPlan_whenDraft_shouldApprove() {
        FulfillmentPlan plan = new FulfillmentPlan();
        plan.setId(1L);
        plan.setStatus(FulfillmentPlanStatus.DRAFT);

        when(fulfillmentPlanRepository.findById(1L)).thenReturn(Optional.of(plan));

        FulfillmentPlan result = domainService.approveFulfillmentPlan(1L);

        assertEquals(FulfillmentPlanStatus.APPROVED, result.getStatus());
    }

    @Test
    void approveFulfillmentPlan_whenNotDraft_shouldThrow() {
        FulfillmentPlan plan = new FulfillmentPlan();
        plan.setId(1L);
        plan.setStatus(FulfillmentPlanStatus.APPROVED);

        when(fulfillmentPlanRepository.findById(1L)).thenReturn(Optional.of(plan));

        assertThrows(IllegalStateException.class, () -> domainService.approveFulfillmentPlan(1L));
    }

    @Test
    void cancelDomOrder_whenCancellable_shouldCancel() {
        DomOrder order = new DomOrder();
        order.setId(1L);
        order.setStatus(DomOrderStatus.PENDING);
        DomOrderLine line = new DomOrderLine();
        line.setStatus(DomOrderLineStatus.PENDING);

        when(domOrderRepository.findById(1L)).thenReturn(Optional.of(order));

        DomOrder result = domainService.cancelDomOrder(1L);

        assertEquals(DomOrderStatus.CANCELLED, result.getStatus());
    }

    @Test
    void cancelDomOrder_whenFulfilled_shouldThrow() {
        DomOrder order = new DomOrder();
        order.setId(1L);
        order.setStatus(DomOrderStatus.FULFILLED);

        when(domOrderRepository.findById(1L)).thenReturn(Optional.of(order));

        assertThrows(IllegalStateException.class, () -> domainService.cancelDomOrder(1L));
    }

    @Test
    void cancelDomOrder_whenAlreadyCancelled_shouldThrow() {
        DomOrder order = new DomOrder();
        order.setId(1L);
        order.setStatus(DomOrderStatus.CANCELLED);

        when(domOrderRepository.findById(1L)).thenReturn(Optional.of(order));

        assertThrows(IllegalStateException.class, () -> domainService.cancelDomOrder(1L));
    }

    @Test
    void saveCancelledOrder_shouldCancelAllLines() {
        DomOrder order = new DomOrder();
        order.setId(1L);
        DomOrderLine line1 = new DomOrderLine();
        line1.setStatus(DomOrderLineStatus.PENDING);
        DomOrderLine line2 = new DomOrderLine();
        line2.setStatus(DomOrderLineStatus.ATP_PASS);

        when(domOrderRepository.save(same(order))).thenReturn(order);
        when(domOrderLineRepository.findByDomOrderId(1L)).thenReturn(List.of(line1, line2));

        domainService.saveCancelledOrder(order);

        assertEquals(DomOrderLineStatus.CANCELLED, line1.getStatus());
        assertEquals(DomOrderLineStatus.CANCELLED, line2.getStatus());
        verify(domOrderLineRepository, times(2)).save(any());
    }
}
