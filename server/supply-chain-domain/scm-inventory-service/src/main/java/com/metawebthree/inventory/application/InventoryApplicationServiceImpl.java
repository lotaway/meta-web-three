package com.metawebthree.inventory.application;

import com.metawebthree.inventory.application.dto.InventoryDTO;
import com.metawebthree.inventory.application.dto.InventoryOperationResult;
import com.metawebthree.inventory.application.dto.ReserveInventoryDTO;
import com.metawebthree.inventory.domain.entity.Inventory;
import com.metawebthree.inventory.domain.entity.InventoryOperationLog;
import com.metawebthree.inventory.domain.entity.ReservationRecord;
import com.metawebthree.inventory.domain.service.InventoryDomainService;
import com.metawebthree.inventory.infrastructure.persistence.repository.InventoryOperationLogRepository;
import com.metawebthree.inventory.infrastructure.persistence.repository.InventoryRepository;
import com.metawebthree.inventory.infrastructure.persistence.repository.ReservationRepository;
import org.springframework.stereotype.Service;
import java.time.LocalDateTime;
import java.util.List;

@Service
public class InventoryApplicationServiceImpl implements InventoryApplicationService {

    private final InventoryDomainService domainService;
    private final InventoryRepository inventoryRepository;
    private final ReservationRepository reservationRepository;
    private final InventoryOperationLogRepository operationLogRepository;

    public InventoryApplicationServiceImpl(InventoryDomainService domainService, 
                                            InventoryRepository inventoryRepository,
                                            ReservationRepository reservationRepository,
                                            InventoryOperationLogRepository operationLogRepository) {
        this.domainService = domainService;
        this.inventoryRepository = inventoryRepository;
        this.reservationRepository = reservationRepository;
        this.operationLogRepository = operationLogRepository;
    }

    @Override
    public InventoryDTO queryBySku(String skuCode, Long warehouseId) {
        return domainService.findBySkuAndWarehouse(skuCode, warehouseId)
            .map(this::toDTO)
            .orElse(null);
    }

    @Override
    public List<InventoryDTO> queryBySkuCode(String skuCode) {
        return inventoryRepository.findBySkuCode(skuCode).stream()
            .map(this::toDTO)
            .collect(java.util.stream.Collectors.toList());
    }

    @Override
    public InventoryOperationResult reserve(ReserveInventoryDTO dto) {
        InventoryOperationLog log = buildOperationLog("RESERVE", dto.getSkuCode(),
            dto.getWarehouseId(), -dto.getQuantity(), dto.getBizId());
        try {
            doReserve(log, dto);
            return InventoryOperationResult.success(dto.getBizId(), dto.getQuantity());
        } catch (Exception e) {
            saveOperationLog(log, null, null, "FAILED", e.getMessage());
            return InventoryOperationResult.fail(e.getMessage());
        }
    }

    @Override
    public InventoryOperationResult confirm(String bizId) {
        InventoryOperationLog log = buildOperationLog("CONFIRM", null, null, null, bizId);
        try {
            var reservation = findReservation(bizId);
            validateReservationStatus(reservation);
            doConfirm(log, reservation);
            return InventoryOperationResult.success(bizId, reservation.getQuantity());
        } catch (Exception e) {
            saveOperationLog(log, null, null, "FAILED", e.getMessage());
            return InventoryOperationResult.fail(e.getMessage());
        }
    }

    @Override
    public InventoryOperationResult cancel(String bizId) {
        InventoryOperationLog log = buildOperationLog("CANCEL", null, null, null, bizId);
        try {
            var reservation = findReservation(bizId);
            validateReservationStatusForCancel(reservation);
            doCancel(log, reservation);
            return InventoryOperationResult.success(bizId, reservation.getQuantity());
        } catch (Exception e) {
            saveOperationLog(log, null, null, "FAILED", e.getMessage());
            return InventoryOperationResult.fail(e.getMessage());
        }
    }

    @Override
    public InventoryOperationResult increase(String skuCode, Long warehouseId,
            Integer quantity, String remark) {
        InventoryOperationLog log = buildOperationLog("INCREASE", skuCode,
            warehouseId, quantity, null);
        log.setRemark(remark);
        try {
            doIncrease(log, skuCode, warehouseId, quantity);
            return InventoryOperationResult.success(null, quantity);
        } catch (Exception e) {
            saveOperationLog(log, null, null, "FAILED", e.getMessage());
            return InventoryOperationResult.fail(e.getMessage());
        }
    }

    @Override
    public InventoryOperationResult decrease(String skuCode, Long warehouseId,
            Integer quantity, String remark) {
        InventoryOperationLog log = buildOperationLog("DECREASE", skuCode,
            warehouseId, -quantity, null);
        log.setRemark(remark);
        try {
            doDecrease(log, skuCode, warehouseId, quantity);
            return InventoryOperationResult.success(null, quantity);
        } catch (Exception e) {
            saveOperationLog(log, null, null, "FAILED", e.getMessage());
            return InventoryOperationResult.fail(e.getMessage());
        }
    }

    // ==================== Private Business Methods ====================

    private void doReserve(InventoryOperationLog log, ReserveInventoryDTO dto) {
        var inventory = domainService.findBySkuAndWarehouse(dto.getSkuCode(), dto.getWarehouseId())
            .orElseGet(() -> domainService.create(dto.getSkuCode(), dto.getWarehouseId()));
        Integer qtyBefore = inventory.getAvailableQuantity();
        domainService.reserve(inventory, dto.getQuantity(), dto.getBizId());
        inventoryRepository.save(inventory);
        Integer qtyAfter = inventory.getAvailableQuantity();
        saveReservationRecord(dto);
        saveOperationLog(log, qtyBefore, qtyAfter, "SUCCESS", null);
    }

    private void doConfirm(InventoryOperationLog log, ReservationRecord reservation) {
        var inventory = domainService.findBySkuAndWarehouse(
            reservation.getSkuCode(), reservation.getWarehouseId())
            .orElseThrow(() -> new IllegalStateException("Inventory not found"));
        Integer qtyBefore = inventory.getTotalQuantity();
        domainService.confirm(inventory, reservation.getQuantity());
        inventoryRepository.save(inventory);
        Integer qtyAfter = inventory.getTotalQuantity();
        reservation.setStatus("CONFIRMED");
        reservationRepository.save(reservation);
        populateLogFromReservation(log, reservation);
        saveOperationLog(log, qtyBefore, qtyAfter, "SUCCESS", null);
    }

    private void doCancel(InventoryOperationLog log, ReservationRecord reservation) {
        var inventory = domainService.findBySkuAndWarehouse(
            reservation.getSkuCode(), reservation.getWarehouseId())
            .orElseThrow(() -> new IllegalStateException("Inventory not found"));
        Integer qtyBefore = inventory.getAvailableQuantity();
        domainService.cancel(inventory, reservation.getQuantity());
        inventoryRepository.save(inventory);
        Integer qtyAfter = inventory.getAvailableQuantity();
        reservation.setStatus("CANCELLED");
        reservationRepository.save(reservation);
        populateLogFromReservation(log, reservation);
        log.setQuantity(reservation.getQuantity());
        saveOperationLog(log, qtyBefore, qtyAfter, "SUCCESS", null);
    }

    private void doDecrease(InventoryOperationLog log, String skuCode, Long warehouseId, Integer quantity) {
        var inventory = domainService.findBySkuAndWarehouse(skuCode, warehouseId)
            .orElseThrow(() -> new IllegalStateException("Inventory not found"));
        Integer qtyBefore = inventory.getTotalQuantity();
        domainService.decrease(inventory, quantity);
        inventoryRepository.save(inventory);
        Integer qtyAfter = inventory.getTotalQuantity();
        saveOperationLog(log, qtyBefore, qtyAfter, "SUCCESS", null);
    }

    private void doIncrease(InventoryOperationLog log, String skuCode, Long warehouseId, Integer quantity) {
        var inventory = domainService.findBySkuAndWarehouse(skuCode, warehouseId)
            .orElseGet(() -> domainService.create(skuCode, warehouseId));
        Integer qtyBefore = inventory.getTotalQuantity();
        domainService.increase(inventory, quantity);
        inventoryRepository.save(inventory);
        Integer qtyAfter = inventory.getTotalQuantity();
        saveOperationLog(log, qtyBefore, qtyAfter, "SUCCESS", null);
    }

    // ==================== Private Helper Methods ====================

    private ReservationRecord findReservation(String bizId) {
        return reservationRepository.findByBizId(bizId)
            .orElseThrow(() -> new IllegalStateException("Reservation not found: " + bizId));
    }

    private InventoryOperationLog buildOperationLog(String opType, String skuCode,
            Long warehouseId, Integer quantity, String bizId) {
        InventoryOperationLog log = new InventoryOperationLog();
        log.setOperationType(opType);
        log.setSkuCode(skuCode);
        log.setWarehouseId(warehouseId);
        log.setQuantity(quantity);
        log.setBizId(bizId);
        return log;
    }

    private void saveOperationLog(InventoryOperationLog log, Integer qtyBefore,
            Integer qtyAfter, String result, String errorMsg) {
        log.setQuantityBefore(qtyBefore);
        log.setQuantityAfter(qtyAfter);
        log.setResult(result);
        log.setErrorMessage(errorMsg);
        log.setOperatedAt(LocalDateTime.now());
        operationLogRepository.save(log);
    }

    private void saveReservationRecord(ReserveInventoryDTO dto) {
        ReservationRecord record = new ReservationRecord();
        record.setBizId(dto.getBizId());
        record.setSkuCode(dto.getSkuCode());
        record.setWarehouseId(dto.getWarehouseId());
        record.setQuantity(dto.getQuantity());
        record.setStatus("PENDING");
        record.setCreatedAt(LocalDateTime.now());
        record.setUpdatedAt(LocalDateTime.now());
        reservationRepository.save(record);
    }

    private void validateReservationStatus(ReservationRecord reservation) {
        if ("CONFIRMED".equals(reservation.getStatus())) {
            throw new IllegalStateException("Reservation already confirmed");
        }
        if ("CANCELLED".equals(reservation.getStatus())) {
            throw new IllegalStateException("Reservation already cancelled");
        }
    }

    private void validateReservationStatusForCancel(ReservationRecord reservation) {
        if ("CONFIRMED".equals(reservation.getStatus())) {
            throw new IllegalStateException("Reservation already confirmed, cannot cancel");
        }
        if ("CANCELLED".equals(reservation.getStatus())) {
            throw new IllegalStateException("Reservation already cancelled");
        }
    }

    private void populateLogFromReservation(InventoryOperationLog log, ReservationRecord reservation) {
        log.setSkuCode(reservation.getSkuCode());
        log.setWarehouseId(reservation.getWarehouseId());
        log.setQuantity(-reservation.getQuantity());
    }

    private InventoryDTO toDTO(Inventory inventory) {
        InventoryDTO dto = new InventoryDTO();
        dto.setId(inventory.getId());
        dto.setSkuCode(inventory.getSkuCode());
        dto.setWarehouseId(inventory.getWarehouseId());
        dto.setTotalQuantity(inventory.getTotalQuantity());
        dto.setAvailableQuantity(inventory.getAvailableQuantity());
        dto.setReservedQuantity(inventory.getReservedQuantity());
        dto.setDefectiveQuantity(inventory.getDefectiveQuantity());
        dto.setUnitCost(inventory.getUnitCost());
        dto.setCreatedAt(inventory.getCreatedAt());
        dto.setUpdatedAt(inventory.getUpdatedAt());
        return dto;
    }
}