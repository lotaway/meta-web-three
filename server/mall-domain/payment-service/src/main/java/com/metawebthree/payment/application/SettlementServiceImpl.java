package com.metawebthree.payment.application;

import com.metawebthree.common.annotations.LogMethod;
import com.metawebthree.payment.domain.model.ExchangeOrder;
import com.metawebthree.payment.infrastructure.persistence.mapper.ExchangeOrderRepository;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.math.BigDecimal;
import java.sql.Timestamp;
import java.time.LocalDate;
import java.util.*;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class SettlementServiceImpl {

    private final ExchangeOrderRepository exchangeOrderRepository;
    private final RestTemplate restTemplate;

    @Value("${payment.settlement.fee-rate:0.002}")
    private BigDecimal feeRate; // Fee rate 0.2%
    
    @Value("${payment.settlement.bank-api-url:}")
    private String bankApiUrl;
    
    @Value("${payment.settlement.payment-platform-api-url:}")
    private String paymentPlatformApiUrl;

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
        if (orders == null || orders.isEmpty()) {
            log.info("No orders to clear for the day");
            return;
        }
        
        Map<String, List<ExchangeOrder>> groupedByUser = orders.stream()
                .collect(Collectors.groupingBy(order -> order.getUserId() != null ? order.getUserId().toString() : "unknown"));
        
        Map<String, List<ExchangeOrder>> groupedByPaymentMethod = orders.stream()
                .collect(Collectors.groupingBy(order -> order.getPaymentMethod() != null ? order.getPaymentMethod().name() : "UNKNOWN"));
        
        log.info("Clearing group statistics - user count: {}, payment method: {}", 
                groupedByUser.size(), groupedByPaymentMethod.size());

        for (Map.Entry<String, List<ExchangeOrder>> entry : groupedByUser.entrySet()) {
            List<ExchangeOrder> userOrders = entry.getValue();
            BigDecimal totalAmount = userOrders.stream()
                    .map(ExchangeOrder::getFiatAmount)
                    .reduce(BigDecimal.ZERO, BigDecimal::add);
            
            BigDecimal totalFee = userOrders.stream()
                    .map(ExchangeOrder::getCryptoAmount)
                    .map(amt -> amt.multiply(feeRate))
                    .reduce(BigDecimal.ZERO, BigDecimal::add);
            
            log.info("User {} clearing amount: {}, fee: {}", 
                    entry.getKey(), totalAmount, totalFee);
        }

        orders.forEach(order -> {
            BigDecimal fee = order.getCryptoAmount().multiply(feeRate);
            order.setFee(fee);
            order.setSettlementAmount(order.getFiatAmount().subtract(fee));
        });
    }

    private void settleOrders(List<ExchangeOrder> orders) {
        if (orders == null || orders.isEmpty()) {
            log.info("No orders to settle for the day");
            return;
        }
        
        Map<String, List<ExchangeOrder>> groupedByUser = orders.stream()
                .collect(Collectors.groupingBy(order -> order.getUserId() != null ? order.getUserId().toString() : "unknown"));
        
        for (Map.Entry<String, List<ExchangeOrder>> entry : groupedByUser.entrySet()) {
            String userId = entry.getKey();
            List<ExchangeOrder> userOrders = entry.getValue();
            
            BigDecimal totalSettlement = userOrders.stream()
                    .map(ExchangeOrder::getSettlementAmount)
                    .reduce(BigDecimal.ZERO, BigDecimal::add);
            
            boolean transferSuccess = executeBankTransfer(userId, totalSettlement, userOrders.size());
            
            if (transferSuccess) {
                userOrders.forEach(order -> {
                    log.info("Order settlement completed - orderNo: {}, settlementAmount: {}", 
                            order.getOrderNo(), order.getSettlementAmount());
                });
            } else {
                log.error("User {} settlement transfer failed, pending retry", userId);
            }
        }
    }
    
    private boolean executeBankTransfer(String userId, BigDecimal amount, int orderCount) {
        if ((bankApiUrl == null || bankApiUrl.isEmpty()) && 
            (paymentPlatformApiUrl == null || paymentPlatformApiUrl.isEmpty())) {
            log.error("Bank API and payment platform API are not configured, cannot execute transfer");
            return false;
        }
        
        try {
            String apiUrl = bankApiUrl != null && !bankApiUrl.isEmpty() ? bankApiUrl : paymentPlatformApiUrl;
            String url = apiUrl + "/settlement/transfer";
            
            Map<String, Object> payload = new HashMap<>();
            payload.put("userId", userId);
            payload.put("amount", amount);
            payload.put("orderCount", orderCount);
            payload.put("settlementDate", LocalDate.now().minusDays(1).toString());
            
            ResponseEntity<Map> response = restTemplate.postForEntity(url, payload, Map.class);
            
            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                Boolean success = (Boolean) response.getBody().get("success");
                if (Boolean.TRUE.equals(success)) {
                    log.info("Bank transfer successful - userId: {}, amount: {}", userId, amount);
                    return true;
                } else {
                    String message = (String) response.getBody().get("message");
                    log.error("Bank transfer failed - userId: {}, error: {}", userId, message);
                    return false;
                }
            }
        } catch (Exception e) {
            log.error("Bank API call exception - userId: {}, error: {}", userId, e.getMessage());
        }
        
        return false;
    }
}