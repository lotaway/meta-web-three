package com.metawebthree.inventory.application;

import com.metawebthree.inventory.application.dto.InventoryOperationResult;
import com.metawebthree.inventory.application.dto.StockInDTO;
import com.metawebthree.inventory.domain.entity.Inventory;
import com.metawebthree.inventory.domain.entity.InventoryOperationLog;
import com.metawebthree.inventory.domain.service.InventoryDomainService;
import com.metawebthree.inventory.infrastructure.persistence.repository.InventoryOperationLogRepository;
import com.metawebthree.inventory.infrastructure.persistence.repository.InventoryRepository;
import com.metawebthree.inventory.infrastructure.persistence.repository.ReservationRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class InventoryStockInIdempotencyTest {

    private InventoryDomainService domainService;
    private InventoryRepository inventoryRepository;
    private InventoryOperationLogRepository operationLogRepository;
    private InventoryApplicationServiceImpl service;

    @BeforeEach
    void setUp() {
        domainService = mock(InventoryDomainService.class);
        inventoryRepository = mock(InventoryRepository.class);
        operationLogRepository = mock(InventoryOperationLogRepository.class);
        service = new InventoryApplicationServiceImpl(
                domainService, inventoryRepository,
                mock(ReservationRepository.class), operationLogRepository);
    }

    private StockInDTO stockIn() {
        StockInDTO dto = new StockInDTO();
        dto.setRmaId(1L);
        dto.setRmaNo("RMA001");
        dto.setSkuCode("SKU001");
        dto.setWarehouseId(100L);
        dto.setQuantity(5);
        return dto;
    }

    private InventoryOperationLog existingLog(String operationType, String result) {
        InventoryOperationLog log = new InventoryOperationLog();
        log.setOperationType(operationType);
        log.setResult(result);
        return log;
    }

    @Test
    void firstStockInIncreasesInventory() {
        String bizId = InventoryApplicationServiceImpl.stockInBizId("RMA001", "SKU001");
        when(operationLogRepository.findByBizId(bizId)).thenReturn(List.of());
        when(domainService.findBySkuAndWarehouse("SKU001", 100L))
                .thenReturn(Optional.of(new Inventory()));

        InventoryOperationResult result = service.stockIn(stockIn());

        assertTrue(result.isSuccess());
        assertFalse(result.isDuplicated());
        verify(domainService).increase(any(Inventory.class), eq(5));
        verify(inventoryRepository).save(any(Inventory.class));
    }

    @Test
    void replayAfterSuccessDoesNotIncreaseAgain() {
        String bizId = InventoryApplicationServiceImpl.stockInBizId("RMA001", "SKU001");
        when(operationLogRepository.findByBizId(bizId))
                .thenReturn(List.of(existingLog("STOCK_IN", "SUCCESS")));

        InventoryOperationResult result = service.stockIn(stockIn());

        assertTrue(result.isSuccess());
        assertTrue(result.isDuplicated());
        verify(domainService, never()).increase(any(Inventory.class), anyInt());
        verify(inventoryRepository, never()).save(any(Inventory.class));
    }

    @Test
    void failedAttemptIsRetryable() {
        String bizId = InventoryApplicationServiceImpl.stockInBizId("RMA001", "SKU001");
        when(operationLogRepository.findByBizId(bizId))
                .thenReturn(List.of(existingLog("STOCK_IN", "FAILED")));
        when(domainService.findBySkuAndWarehouse("SKU001", 100L))
                .thenReturn(Optional.of(new Inventory()));

        InventoryOperationResult result = service.stockIn(stockIn());

        assertTrue(result.isSuccess());
        assertFalse(result.isDuplicated());
        verify(domainService).increase(any(Inventory.class), eq(5));
    }

    @Test
    void rejectsNonPositiveQuantity() {
        StockInDTO dto = stockIn();
        dto.setQuantity(0);

        InventoryOperationResult result = service.stockIn(dto);

        assertFalse(result.isSuccess());
        verify(domainService, never()).increase(any(Inventory.class), anyInt());
    }

    @Test
    void rejectsMissingSku() {
        StockInDTO dto = stockIn();
        dto.setSkuCode(" ");

        InventoryOperationResult result = service.stockIn(dto);

        assertFalse(result.isSuccess());
        verify(domainService, never()).increase(any(Inventory.class), anyInt());
    }
}
