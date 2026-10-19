package com.metawebthree.rma.application;

import com.metawebthree.rma.application.integration.InventoryIntegrationService;
import com.metawebthree.rma.application.integration.StockInResult;
import com.metawebthree.rma.domain.entity.RmaStockInRecord;
import com.metawebthree.rma.domain.repository.RmaStockInRecordRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Coordinates the return-to-stock step of an RMA.
 *
 * <p>Every accepted (inspection-passed) line is tracked in a stock-in ledger
 * keyed by {@code rmaId + skuCode}. A line that already succeeded is never sent
 * to inventory again, and a failed line is recorded (with its error) so the
 * scheduled retry job can deliver it later.</p>
 */
@Slf4j
@Service
public class RmaStockInService {

    public static final String STATUS_PENDING = "PENDING";
    public static final String STATUS_SUCCESS = "SUCCESS";
    public static final String STATUS_FAILED = "FAILED";

    private static final int MAX_ERROR_LENGTH = 500;

    private final RmaStockInRecordRepository rmaStockInRecordRepository;
    private final InventoryIntegrationService inventoryIntegrationService;

    public RmaStockInService(RmaStockInRecordRepository rmaStockInRecordRepository,
                             InventoryIntegrationService inventoryIntegrationService) {
        this.rmaStockInRecordRepository = rmaStockInRecordRepository;
        this.inventoryIntegrationService = inventoryIntegrationService;
    }

    /**
     * Register an accepted RMA line and try to deliver it to the inventory
     * service. Idempotent: replaying the same line after a success is a no-op.
     */
    public void dispatch(Long rmaId, String rmaNo, String skuCode, String skuName,
                         Integer quantity, Long warehouseId) {
        if (rmaId == null || skuCode == null || skuCode.isBlank()) {
            log.warn("Skip RMA stock-in - missing rmaId/skuCode (rmaId={}, skuCode={})", rmaId, skuCode);
            return;
        }
        if (quantity == null || quantity <= 0) {
            log.warn("Skip RMA stock-in - non-positive quantity (rmaId={}, skuCode={}, quantity={})",
                    rmaId, skuCode, quantity);
            return;
        }

        RmaStockInRecord record = rmaStockInRecordRepository
                .findByRmaIdAndSkuCode(rmaId, skuCode)
                .orElseGet(RmaStockInRecord::new);

        if (STATUS_SUCCESS.equals(record.getStatus())) {
            log.info("RMA stock-in already succeeded, skip replay - rmaId={}, skuCode={}", rmaId, skuCode);
            return;
        }

        record.setRmaId(rmaId);
        record.setRmaNo(rmaNo);
        record.setSkuCode(skuCode);
        record.setSkuName(skuName);
        record.setWarehouseId(warehouseId);
        record.setQuantity(quantity);
        record.setStatus(STATUS_PENDING);
        if (record.getRetryCount() == null) {
            record.setRetryCount(0);
        }
        if (record.getCreatedAt() == null) {
            record.setCreatedAt(LocalDateTime.now());
        }
        record.setUpdatedAt(LocalDateTime.now());
        rmaStockInRecordRepository.save(record);

        deliver(record);
    }

    /**
     * Re-deliver pending and previously failed lines. Lines that exhausted
     * {@code maxRetries} are left untouched for manual reconciliation.
     *
     * @return the number of records that were retried
     */
    public int retryPending(int maxRetries) {
        List<RmaStockInRecord> pending = rmaStockInRecordRepository
                .findByStatusIn(List.of(STATUS_PENDING, STATUS_FAILED));
        int retried = 0;
        for (RmaStockInRecord record : pending) {
            int attempts = record.getRetryCount() != null ? record.getRetryCount() : 0;
            if (attempts >= maxRetries) {
                log.warn("RMA stock-in exhausted retries - rmaId={}, skuCode={}, attempts={}",
                        record.getRmaId(), record.getSkuCode(), attempts);
                continue;
            }
            deliver(record);
            retried++;
        }
        return retried;
    }

    private void deliver(RmaStockInRecord record) {
        StockInResult result;
        try {
            result = inventoryIntegrationService.stockIn(
                    record.getRmaId(), record.getRmaNo(), record.getSkuCode(),
                    record.getSkuName(), record.getQuantity(), record.getWarehouseId());
        } catch (Exception e) {
            log.error("RMA stock-in threw unexpectedly - rmaId={}, skuCode={}",
                    record.getRmaId(), record.getSkuCode(), e);
            result = StockInResult.failure(e.getMessage());
        }

        if (result.isSuccess()) {
            record.setStatus(STATUS_SUCCESS);
            record.setLastError(null);
            log.info("RMA stock-in recorded - rmaId={}, skuCode={}, quantity={}, duplicated={}",
                    record.getRmaId(), record.getSkuCode(), record.getQuantity(), result.isDuplicated());
        } else {
            record.setStatus(STATUS_FAILED);
            record.setLastError(truncate(result.getMessage()));
            record.setRetryCount((record.getRetryCount() != null ? record.getRetryCount() : 0) + 1);
            log.warn("RMA stock-in failed, will retry - rmaId={}, skuCode={}, error={}",
                    record.getRmaId(), record.getSkuCode(), record.getLastError());
        }
        record.setUpdatedAt(LocalDateTime.now());
        rmaStockInRecordRepository.save(record);
    }

    private String truncate(String message) {
        if (message == null) {
            return null;
        }
        return message.length() <= MAX_ERROR_LENGTH ? message : message.substring(0, MAX_ERROR_LENGTH);
    }
}
