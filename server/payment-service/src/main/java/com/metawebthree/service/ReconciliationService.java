package com.metawebthree.service;

import com.metawebthree.common.annotations.LogMethod;
import com.metawebthree.entity.ExchangeOrder;
import com.metawebthree.repository.ExchangeOrderRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.sql.Timestamp;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;

/**
 * Reconciliation service
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class ReconciliationService {

    private final ExchangeOrderRepository exchangeOrderRepository;

    @Scheduled(cron = "0 0 1 * * ?")
    @LogMethod
    public void dailyReconciliation() {
        LocalDate reconciliationDate = LocalDate.now().minusDays(1);
        List<ExchangeOrder> internalOrders = getInternalOrders(reconciliationDate);
        List<ExchangeOrder> externalBills = getExternalBills(reconciliationDate);
        reconcileOrders(internalOrders, externalBills);
    }

    private List<ExchangeOrder> getInternalOrders(LocalDate date) {
        Timestamp start = Timestamp.valueOf(date.atStartOfDay());
        Timestamp end = Timestamp.valueOf(date.plusDays(1).atStartOfDay());
        return exchangeOrderRepository.findByCreatedAtBetween(start, end);
    }

    private List<ExchangeOrder> getExternalBills(LocalDate date) {
        // TODO: Should call payment platform API to get bills
        return List.of();
    }

    /**
     * Execute reconciliation
     */
    private void reconcileOrders(List<ExchangeOrder> internalOrders, List<ExchangeOrder> externalBills) {
        // 1. Check missing orders (exist externally but not internally)
        checkMissingOrders(internalOrders, externalBills);
        
        // 2. Check extra orders (exist internally but not externally)
        checkExtraOrders(internalOrders, externalBills);
        
        // 3. Check amount mismatches
        checkAmountMismatches(internalOrders, externalBills);
    }

    /**
     * Check missing orders
     */
    private void checkMissingOrders(List<ExchangeOrder> internalOrders, List<ExchangeOrder> externalBills) {
        // TODO: Implement missing order check logic
        log.info("Checking for missing orders...");
    }

    /**
     * Check extra orders
     */
    private void checkExtraOrders(List<ExchangeOrder> internalOrders, List<ExchangeOrder> externalBills) {
        // TODO: Implement extra order check logic
        log.info("Checking for extra orders...");
    }

    /**
     * Check amount mismatches
     */
    private void checkAmountMismatches(List<ExchangeOrder> internalOrders, List<ExchangeOrder> externalBills) {
        // TODO: Implement amount mismatch check logic
        log.info("Checking for amount mismatches...");
    }

    /**
     * Manual reconciliation trigger
     */
    public void manualReconciliation(LocalDate date) {
        log.info("Manual reconciliation triggered for date: {}", date);
        dailyReconciliation();
    }
}
