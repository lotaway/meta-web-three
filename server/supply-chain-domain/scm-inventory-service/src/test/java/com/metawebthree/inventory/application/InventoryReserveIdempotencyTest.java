package com.metawebthree.inventory.application;

import com.metawebthree.inventory.application.dto.InventoryOperationResult;
import com.metawebthree.inventory.application.dto.ReserveInventoryDTO;
import com.metawebthree.inventory.domain.entity.Inventory;
import com.metawebthree.inventory.domain.entity.ReservationRecord;
import com.metawebthree.inventory.domain.service.InventoryDomainService;
import com.metawebthree.inventory.infrastructure.persistence.repository.InventoryOperationLogRepository;
import com.metawebthree.inventory.infrastructure.persistence.repository.InventoryRepository;
import com.metawebthree.inventory.infrastructure.persistence.repository.ReservationRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class InventoryReserveIdempotencyTest {

    private InventoryDomainService domainService;
    private InventoryRepository inventoryRepository;
    private ReservationRepository reservationRepository;
    private InventoryApplicationServiceImpl service;

    @BeforeEach
    void setUp() {
        domainService = mock(InventoryDomainService.class);
        inventoryRepository = mock(InventoryRepository.class);
        reservationRepository = mock(ReservationRepository.class);
        service = new InventoryApplicationServiceImpl(domainService, inventoryRepository,
                reservationRepository, mock(InventoryOperationLogRepository.class));
    }

    private ReserveInventoryDTO dto() {
        ReserveInventoryDTO dto = new ReserveInventoryDTO();
        dto.setSkuCode("9");
        dto.setWarehouseId(1L);
        dto.setQuantity(2);
        dto.setBizId("ORDER-100");
        return dto;
    }

    @Test
    void duplicateReserveIsIdempotentAndDoesNotTouchInventory() {
        ReservationRecord existing = new ReservationRecord();
        existing.setBizId("ORDER-100");
        existing.setQuantity(2);
        when(reservationRepository.findByBizId("ORDER-100")).thenReturn(Optional.of(existing));

        InventoryOperationResult result = service.reserve(dto());

        assertTrue(result.isSuccess());
        assertEquals(2, result.getQuantity());
        verify(domainService, never()).reserve(any(), anyInt(), anyString());
        verify(inventoryRepository, never()).save(any());
        verify(reservationRepository, never()).save(any());
    }

    @Test
    void firstReserveCreatesReservationRecord() {
        when(reservationRepository.findByBizId("ORDER-100")).thenReturn(Optional.empty());
        when(domainService.findBySkuAndWarehouse("9", 1L)).thenReturn(Optional.of(mock(Inventory.class)));

        InventoryOperationResult result = service.reserve(dto());

        assertTrue(result.isSuccess());
        verify(domainService).reserve(any(), eq(2), eq("ORDER-100"));
        verify(inventoryRepository).save(any());
        verify(reservationRepository).save(any());
    }
}
