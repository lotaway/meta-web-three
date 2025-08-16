package com.metawebthree.service;

import com.metawebthree.entity.ExchangeOrder;
import com.metawebthree.repository.ExchangeOrderRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;

/**
 * Clearing and settlement service
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class SettlementService {

    private final ExchangeOrderRepository exchangeOrderRepository;
    private final ReconciliationService reconciliationService;

    @Value("${payment.settlement.fee-rate:0.002}")
    private BigDecimal feeRate; // Fee rate 0.2%

    /**
     * Daily clearing task (runs at 2 AM)
     */
    @Scheduled(cron = "0 0 2 * * ?")
    public void dailyClearing() {
        LocalDate settlementDate = LocalDate.now().minusDays(1);
        log.info("Starting daily clearing for date: {}", settlementDate);

        // 1. Run reconciliation first
        reconciliationService.manualReconciliation(settlementDate);

        // 2. Execute clearing
        List<ExchangeOrder> orders = getSettlementOrders(settlementDate);
        clearingOrders(orders);

        log.info("Daily clearing completed for date: {}", settlementDate);
    }

    /**
     * Get pending settlement orders
     */
    private List<ExchangeOrder> getSettlementOrders(LocalDate date) {
        LocalDateTime start = date.atStartOfDay();
        LocalDateTime end = date.plusDays(1).atStartOfDay();
        return exchangeOrderRepository.findByStatusAndCreatedAtBetween(
            "SUCCESS", start, end);
    }

    /**
     * Order clearing
     */
    private void clearingOrders(List<ExchangeOrder> orders) {
        // Group by merchant/channel
        // TODO: Implement grouping logic
        
        // Calculate fees
        orders.forEach(order -> {
            BigDecimal fee = order.getAmount().multiply(feeRate);
            order.setFee(fee);
            order.setSettlementAmount(order.getAmount().subtract(fee));
        });

        log.info("Clearing completed for {} orders", orders.size());
    }

    /**
     * Settlement execution (T+1)
     */
    @Scheduled(cron = "0 0 9 * * ?")
    public void executeSettlement() {
        LocalDate settlementDate = LocalDate.now().minusDays(1);
        log.info("Executing settlement for date: {}", settlementDate);

        // 1. Get cleared orders
        List<ExchangeOrder> orders = getSettlementOrders(settlementDate);

        // 2. Execute settlement
        settleOrders(orders);

        log.info("Settlement completed for date: {}", settlementDate);
    }

    /**
     * Order settlement
     */
    private void settleOrders(List<ExchangeOrder> orders) {
        // TODO: Call bank/payment platform API for transfer
        orders.forEach(order -> {
            log.info("Settling order {}: amount={}, fee={}, settlementAmount={}",
                order.getOrderNo(), order.getAmount(), order.getFee(), 
                order.getSettlementAmount());
        });
    }

    /**
     * Manual settlement trigger
     */
    public void manualSettlement(LocalDate date) {
        log.info("Manual settlement triggered for date: {}", date);
        executeSettlement();
    }
}
