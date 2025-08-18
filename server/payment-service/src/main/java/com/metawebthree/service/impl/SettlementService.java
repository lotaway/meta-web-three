package com.metawebthree.service.impl;

import com.metawebthree.common.annotations.LogMethod;
import com.metawebthree.entity.ExchangeOrder;
import com.metawebthree.repository.ExchangeOrderRepository;
import com.metawebthree.service.impl.ReconciliationServiceImpl;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import org.springframework.beans.factory.annotation.Value;
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
    private final ReconciliationServiceImpl reconciliationService;

    @Value("${payment.settlement.fee-rate:0.002}")
    private BigDecimal feeRate; // Fee rate 0.2%

    /**
     * Daily clearing task (runs at 2 AM)
     */
    @Scheduled(cron = "0 0 2 * * ?")
    @LogMethod
    public void dailyClearing() {
        LocalDate settlementDate = LocalDate.now().minusDays(1);
        reconciliationService.manualReconciliation(settlementDate);
        List<ExchangeOrder> orders = getSettlementOrders(settlementDate);
        clearingOrders(orders);
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

    @LogMethod
    private void clearingOrders(List<ExchangeOrder> orders) {
        // Group by merchant/channel
        // TODO: Implement grouping logic

        // Calculate fees
        orders.forEach(order -> {
            BigDecimal fee = order.getAmount().multiply(feeRate);
            order.setFee(fee);
            order.setSettlementAmount(order.getAmount().subtract(fee));
        });
    }

    @Scheduled(cron = "0 0 9 * * ?")
    @LogMethod
    public void executeSettlement() {
        LocalDate settlementDate = LocalDate.now().minusDays(1);
        List<ExchangeOrder> orders = getSettlementOrders(settlementDate);
        settleOrders(orders);
    }

    private void settleOrders(List<ExchangeOrder> orders) {
        // TODO: Call bank/payment platform API for transfer
        orders.forEach(order -> {
            log.info("Settling order {}: amount={}, fee={}, settlementAmount={}",
                    order.getOrderNo(),
                    order.getCryptoAmount(),
                    order.getFee(),
                    order.getFiatAmount());
        });
    }

    @LogMethod
    public void manualSettlement(LocalDate date) {
        executeSettlement();
    }
}
