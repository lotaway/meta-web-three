package com.metawebthree.rma.application;

import com.metawebthree.rma.application.integration.InventoryIntegrationService;
import com.metawebthree.rma.application.integration.StockInResult;
import com.metawebthree.rma.domain.entity.RmaStockInRecord;
import com.metawebthree.rma.domain.repository.RmaStockInRecordRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class RmaStockInServiceTest {

    private RmaStockInRecordRepository repository;
    private InventoryIntegrationService inventoryIntegrationService;
    private RmaStockInService service;

    @BeforeEach
    void setUp() {
        repository = mock(RmaStockInRecordRepository.class);
        inventoryIntegrationService = mock(InventoryIntegrationService.class);
        service = new RmaStockInService(repository, inventoryIntegrationService);
        when(repository.save(any(RmaStockInRecord.class))).thenAnswer(inv -> inv.getArgument(0));
    }

    private RmaStockInRecord lastSavedRecord(int expectedSaves) {
        ArgumentCaptor<RmaStockInRecord> captor = ArgumentCaptor.forClass(RmaStockInRecord.class);
        verify(repository, times(expectedSaves)).save(captor.capture());
        return captor.getAllValues().get(captor.getAllValues().size() - 1);
    }

    @Test
    void dispatchPersistsSuccess() {
        when(repository.findByRmaIdAndSkuCode(1L, "SKU001")).thenReturn(Optional.empty());
        when(inventoryIntegrationService.stockIn(eq(1L), eq("RMA001"), eq("SKU001"),
                eq("Widget"), eq(3), eq(100L))).thenReturn(StockInResult.success(false));

        service.dispatch(1L, "RMA001", "SKU001", "Widget", 3, 100L);

        RmaStockInRecord saved = lastSavedRecord(2);
        assertEquals("SUCCESS", saved.getStatus());
        assertEquals("SKU001", saved.getSkuCode());
        assertEquals(3, saved.getQuantity());
        verify(inventoryIntegrationService, times(1))
                .stockIn(1L, "RMA001", "SKU001", "Widget", 3, 100L);
    }

    @Test
    void replayAfterSuccessDoesNotCallInventoryAgain() {
        RmaStockInRecord existing = new RmaStockInRecord();
        existing.setId(9L);
        existing.setRmaId(1L);
        existing.setSkuCode("SKU001");
        existing.setStatus("SUCCESS");
        when(repository.findByRmaIdAndSkuCode(1L, "SKU001")).thenReturn(Optional.of(existing));

        service.dispatch(1L, "RMA001", "SKU001", "Widget", 3, 100L);

        verify(inventoryIntegrationService, never())
                .stockIn(any(), any(), any(), any(), anyInt(), any());
        verify(repository, never()).save(any(RmaStockInRecord.class));
    }

    @Test
    void dispatchRecordsFailureForRetry() {
        when(repository.findByRmaIdAndSkuCode(1L, "SKU001")).thenReturn(Optional.empty());
        when(inventoryIntegrationService.stockIn(any(), any(), any(), any(), anyInt(), any()))
                .thenReturn(StockInResult.failure("inventory down"));

        service.dispatch(1L, "RMA001", "SKU001", "Widget", 3, 100L);

        RmaStockInRecord saved = lastSavedRecord(2);
        assertEquals("FAILED", saved.getStatus());
        assertEquals(1, saved.getRetryCount());
        assertEquals("inventory down", saved.getLastError());
    }

    @Test
    void retryPendingRedeliversFailedRecords() {
        RmaStockInRecord failed = new RmaStockInRecord();
        failed.setId(5L);
        failed.setRmaId(1L);
        failed.setRmaNo("RMA001");
        failed.setSkuCode("SKU001");
        failed.setSkuName("Widget");
        failed.setWarehouseId(100L);
        failed.setQuantity(3);
        failed.setStatus("FAILED");
        failed.setRetryCount(1);
        when(repository.findByStatusIn(List.of("PENDING", "FAILED"))).thenReturn(List.of(failed));
        when(inventoryIntegrationService.stockIn(1L, "RMA001", "SKU001", "Widget", 3, 100L))
                .thenReturn(StockInResult.success(false));

        int retried = service.retryPending(10);

        assertEquals(1, retried);
        assertEquals("SUCCESS", failed.getStatus());
        verify(inventoryIntegrationService, times(1))
                .stockIn(1L, "RMA001", "SKU001", "Widget", 3, 100L);
    }

    @Test
    void retryPendingSkipsExhaustedRecords() {
        RmaStockInRecord failed = new RmaStockInRecord();
        failed.setId(5L);
        failed.setRmaId(1L);
        failed.setSkuCode("SKU001");
        failed.setStatus("FAILED");
        failed.setRetryCount(10);
        when(repository.findByStatusIn(any())).thenReturn(List.of(failed));

        int retried = service.retryPending(10);

        assertEquals(0, retried);
        verify(inventoryIntegrationService, never())
                .stockIn(any(), any(), any(), any(), anyInt(), any());
    }

    @Test
    void dispatchSkipsNonPositiveQuantityAndMissingSku() {
        service.dispatch(1L, "RMA001", "SKU001", "Widget", 0, 100L);
        service.dispatch(1L, "RMA001", " ", "Widget", 3, 100L);

        verify(inventoryIntegrationService, never())
                .stockIn(any(), any(), any(), any(), anyInt(), any());
        verify(repository, never()).save(any(RmaStockInRecord.class));
    }

    @Test
    void dispatchTreatsIntegrationExceptionAsFailure() {
        when(repository.findByRmaIdAndSkuCode(1L, "SKU001")).thenReturn(Optional.empty());
        when(inventoryIntegrationService.stockIn(any(), any(), any(), any(), anyInt(), any()))
                .thenThrow(new RuntimeException("timeout"));

        service.dispatch(1L, "RMA001", "SKU001", "Widget", 3, 100L);

        RmaStockInRecord saved = lastSavedRecord(2);
        assertEquals("FAILED", saved.getStatus());
        assertTrue(saved.getLastError().contains("timeout"));
    }
}
