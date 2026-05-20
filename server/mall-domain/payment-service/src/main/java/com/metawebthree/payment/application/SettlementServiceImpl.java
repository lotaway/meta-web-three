package com.metawebthree.payment.application;
 
import com.metawebthree.common.annotations.LogMethod;
import com.metawebthree.payment.domain.model.ExchangeOrder;
import com.metawebthree.payment.infrastructure.persistence.mapper.ExchangeOrderRepository;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.sql.Timestamp;
import java.time.LocalDate;
import java.util.List;

@Service
@RequiredArgsConstructor
@Slf4j
public class SettlementServiceImpl {

    private final ExchangeOrderRepository exchangeOrderRepository;

    @Value("${payment.settlement.fee-rate:0.002}")
    private BigDecimal feeRate; // Fee rate 0.2%

    private List<ExchangeOrder> getSettlementOrders(LocalDate date) {
        Timestamp start = Timestamp.valueOf(date.atStartOfDay());
        Timestamp end = Timestamp.valueOf(date.plusDays(1).atStartOfDay());
        return exchangeOrderRepository.findByStatusAndCreatedAtBetween(
                "SUCCESS", start, end);
    }

    @LogMethod
    public void dailyClearing(String from) {
        LocalDate settlementDate = LocalDate.now().minusDays(1);
        List<ExchangeOrder> orders = getSettlementOrders(settlementDate);
        clearingOrders(orders);
    }

    @LogMethod
    public void executeSettlement(String from) {
        LocalDate settlementDate = LocalDate.now().minusDays(1);
        List<ExchangeOrder> orders = getSettlementOrders(settlementDate);
        settleOrders(orders);
    }

    @LogMethod
    private void clearingOrders(List<ExchangeOrder> orders) {
        // Group by merchant/channel
        // @TODO: Implement grouping logic

        // Calculate fees
        orders.forEach(order -> {
            BigDecimal fee = order.getCryptoAmount().multiply(feeRate);
            order.setFee(fee);
            order.setSettlementAmount(order.getFiatAmount().subtract(fee));
        });
    }

    private void settleOrders(List<ExchangeOrder> orders) {
        // @TODO: Call bank/payment platform API for transfer
        orders.forEach(order -> {
            log.info("Settling order {}: amount={}, fee={}, settlementAmount={}",
                    order.getOrderNo(),
                    order.getCryptoAmount(),
                    order.getFee(),
                    order.getFiatAmount());
        });
    }
}
