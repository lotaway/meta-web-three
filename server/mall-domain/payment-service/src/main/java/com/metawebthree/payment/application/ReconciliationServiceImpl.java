package com.metawebthree.payment.application;

import com.metawebthree.common.annotations.LogMethod;
import com.metawebthree.payment.domain.model.ExchangeOrder;
import com.metawebthree.payment.infrastructure.persistence.dataobject.ReconciliationDiffDO;
import com.metawebthree.payment.infrastructure.persistence.mapper.ExchangeOrderRepository;
import com.metawebthree.payment.domain.repository.ReconciliationDiffRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.quartz.Job;
import org.quartz.JobExecutionContext;
import org.quartz.JobExecutionException;
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
public class ReconciliationServiceImpl {

    private final ExchangeOrderRepository exchangeOrderRepository;
    private final ReconciliationDiffRepository reconciliationDiffRepository;
    private final RestTemplate restTemplate;

    @Value("${reconciliation.payment-platform-api-url:}")
    private String paymentPlatformApiUrl;
    
    @Value("${reconciliation.message-service-url:}")
    private String messageServiceUrl;
    
    @Value("${reconciliation.dingtalk-webhook-url:}")
    private String dingTalkWebhookUrl;

    public static class DailyReconciliationJob implements Job {
        private ReconciliationServiceImpl service;

        public void setService(ReconciliationServiceImpl service) {
            this.service = service;
        }

        @Override
        public void execute(JobExecutionContext context) throws JobExecutionException {
            log.info("Quartz reconciliation job started");
            try {
                service.executeDailyReconciliation();
            } catch (Exception e) {
                log.error("Quartz reconciliation job failed", e);
                throw new JobExecutionException(e);
            }
        }
    }

    @LogMethod
    public void executeDailyReconciliation() {
        log.info("Starting daily reconciliation task");
        LocalDate reconciliationDate = LocalDate.now().minusDays(1);
        List<ExchangeOrder> internalOrders = fetchInternalOrders(reconciliationDate);
        List<ExchangeOrder> externalBills = fetchExternalBills(reconciliationDate);
        performReconciliation(internalOrders, externalBills);
        log.info("Daily reconciliation task completed");
    }

    @Deprecated
    public void dailyReconciliation() {
        executeDailyReconciliation();
    }

    private List<ExchangeOrder> fetchInternalOrders(LocalDate date) {
        Timestamp start = Timestamp.valueOf(date.atStartOfDay());
        Timestamp end = Timestamp.valueOf(date.plusDays(1).atStartOfDay());
        return exchangeOrderRepository.findByCreatedAtBetween(start, end);
    }

    private List<ExchangeOrder> fetchExternalBills(LocalDate date) {
        if (paymentPlatformApiUrl == null || paymentPlatformApiUrl.isEmpty()) {
            log.error("Payment platform API URL not configured, cannot fetch external bills");
            return List.of();
        }
        try {
            String url = paymentPlatformApiUrl + "/bills?date=" + date;
            log.info("Fetching external bills from payment platform - URL: {}", url);
            ResponseEntity<List> response = restTemplate.getForEntity(url, List.class);
            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                return mapToExchangeOrders(response.getBody());
            }
        } catch (Exception e) {
            log.error("Failed to call payment platform API: {}", e.getMessage());
        }
        return List.of();
    }

    @SuppressWarnings("unchecked")
    private List<ExchangeOrder> mapToExchangeOrders(List<Map<String, Object>> bills) {
        if (bills == null || bills.isEmpty()) {
            return List.of();
        }
        return bills.stream()
            .map(bill -> {
                ExchangeOrder order = new ExchangeOrder();
                order.setOrderNo((String) bill.get("orderNo"));
                order.setFiatAmount(new BigDecimal(bill.get("amount").toString()));
                order.setStatus(ExchangeOrder.OrderStatus.valueOf((String) bill.get("status")));
                return order;
            })
            .collect(Collectors.toList());
    }

    private void performReconciliation(List<ExchangeOrder> internalOrders, List<ExchangeOrder> externalBills) {
        detectMissingOrders(internalOrders, externalBills);
        detectExtraOrders(internalOrders, externalBills);
        detectAmountMismatches(internalOrders, externalBills);
    }

    private void detectMissingOrders(List<ExchangeOrder> internalOrders, List<ExchangeOrder> externalBills) {
        if (externalBills.isEmpty() && internalOrders.isEmpty()) {
            log.info("No orders for reconciliation date, skipping check");
            return;
        }

        Set<String> internalOrderNos = internalOrders.stream()
            .map(ExchangeOrder::getOrderNo)
            .collect(Collectors.toSet());

        List<ReconciliationDiffDO> missingOrderDiffs = externalBills.stream()
            .filter(bill -> !internalOrderNos.contains(bill.getOrderNo()))
            .map(bill -> createDiff(ReconciliationDiffDO.DiffType.MISSING_ORDER, bill.getOrderNo(), 
                BigDecimal.ZERO, bill.getFiatAmount(), bill.getFiatAmount()))
            .collect(Collectors.toList());

        if (!missingOrderDiffs.isEmpty()) {
            reconciliationDiffRepository.saveBatch(missingOrderDiffs);
            log.info("Detected {} missing orders, recorded to reconciliation diff table", missingOrderDiffs.size());
            triggerAlert("MISSING_ORDER", missingOrderDiffs.size());
        } else {
            log.info("No missing orders detected");
        }
    }

    private void detectExtraOrders(List<ExchangeOrder> internalOrders, List<ExchangeOrder> externalBills) {
        if (externalBills.isEmpty() && internalOrders.isEmpty()) {
            log.info("No orders for reconciliation date, skipping check");
            return;
        }

        Set<String> externalOrderNos = externalBills.stream()
            .map(ExchangeOrder::getOrderNo)
            .collect(Collectors.toSet());

        List<ReconciliationDiffDO> extraOrderDiffs = internalOrders.stream()
            .filter(order -> !externalOrderNos.contains(order.getOrderNo()))
            .map(order -> createDiff(ReconciliationDiffDO.DiffType.EXTRA_ORDER, order.getOrderNo(),
                order.getFiatAmount(), BigDecimal.ZERO, order.getFiatAmount().negate()))
            .collect(Collectors.toList());

        if (!extraOrderDiffs.isEmpty()) {
            reconciliationDiffRepository.saveBatch(extraOrderDiffs);
            log.info("Detected {} extra orders, recorded to reconciliation diff table", extraOrderDiffs.size());
            triggerAlert("EXTRA_ORDER", extraOrderDiffs.size());
        } else {
            log.info("No extra orders detected");
        }
    }

    private void detectAmountMismatches(List<ExchangeOrder> internalOrders, List<ExchangeOrder> externalBills) {
        if (internalOrders.isEmpty() || externalBills.isEmpty()) {
            log.info("Internal or external orders empty, skipping amount check");
            return;
        }

        Map<String, ExchangeOrder> internalOrderMap = internalOrders.stream()
            .collect(Collectors.toMap(ExchangeOrder::getOrderNo, o -> o, (a, b) -> a));

        List<ReconciliationDiffDO> amountMismatchDiffs = externalBills.stream()
            .filter(bill -> internalOrderMap.containsKey(bill.getOrderNo()))
            .filter(bill -> {
                ExchangeOrder internalOrder = internalOrderMap.get(bill.getOrderNo());
                return internalOrder.getFiatAmount().compareTo(bill.getFiatAmount()) != 0;
            })
            .map(bill -> {
                ExchangeOrder internalOrder = internalOrderMap.get(bill.getOrderNo());
                BigDecimal difference = bill.getFiatAmount().subtract(internalOrder.getFiatAmount());
                return createDiff(ReconciliationDiffDO.DiffType.AMOUNT_MISMATCH, bill.getOrderNo(),
                    internalOrder.getFiatAmount(), bill.getFiatAmount(), difference);
            })
            .collect(Collectors.toList());

        if (!amountMismatchDiffs.isEmpty()) {
            reconciliationDiffRepository.saveBatch(amountMismatchDiffs);
            log.info("Detected {} amount mismatch orders, recorded to reconciliation diff table", amountMismatchDiffs.size());
            triggerAlert("AMOUNT_MISMATCH", amountMismatchDiffs.size());
        } else {
            log.info("No amount mismatch orders detected");
        }
    }

    private ReconciliationDiffDO createDiff(ReconciliationDiffDO.DiffType type, String orderNo,
            BigDecimal internalAmount, BigDecimal externalAmount, BigDecimal amountDiff) {
        return ReconciliationDiffDO.builder()
            .reconciliationDate(LocalDate.now().minusDays(1))
            .diffType(type)
            .orderNo(orderNo)
            .internalAmount(internalAmount)
            .externalAmount(externalAmount)
            .amountDifference(amountDiff)
            .status(ReconciliationDiffDO.DiffStatus.PENDING)
            .build();
    }

    private void triggerAlert(String diffType, long count) {
        String title = "Reconciliation Alert";
        String content = String.format("Detected %s alert, count: %d, date: %s", 
            getDiffTypeLabel(diffType), count, LocalDate.now().minusDays(1));
        
        sendInAppNotification(title, content, diffType);
        sendDingTalkNotification(title, content, diffType);
    }
    
    private String getDiffTypeLabel(String diffType) {
        return switch (diffType) {
            case "MISSING_ORDER" -> "Missing Order (external exists, internal missing)";
            case "EXTRA_ORDER" -> "Extra Order (internal exists, external missing)";
            case "AMOUNT_MISMATCH" -> "Amount Mismatch";
            default -> "Unknown";
        };
    }
    
    private void sendInAppNotification(String title, String content, String diffType) {
        if (messageServiceUrl == null || messageServiceUrl.isEmpty()) {
            log.warn("Message service URL not configured, cannot send in-app notification");
            return;
        }
        try {
            String url = messageServiceUrl + "/notification/send";
            Map<String, Object> payload = new HashMap<>();
            payload.put("title", title);
            payload.put("content", content);
            payload.put("type", "RECONCILIATION_ALERT");
            payload.put("relatedId", diffType);
            payload.put("icon", "warning");
            restTemplate.postForEntity(url, payload, Void.class);
            log.info("In-app notification sent successfully: {}", title);
        } catch (Exception e) {
            log.error("In-app notification failed: {}", e.getMessage());
        }
    }
    
    private void sendDingTalkNotification(String title, String content, String diffType) {
        if (dingTalkWebhookUrl == null || dingTalkWebhookUrl.isEmpty()) {
            return;
        }
        try {
            Map<String, Object> payload = new HashMap<>();
            payload.put("msgtype", "text");
            Map<String, Object> text = new HashMap<>();
            text.put("content", String.format("[%s] %s\n%s", title, content, "Please handle promptly"));
            payload.put("text", text);
            restTemplate.postForEntity(dingTalkWebhookUrl, payload, Void.class);
            log.info("DingTalk notification sent successfully: {}", title);
        } catch (Exception e) {
            log.error("DingTalk notification failed: {}", e.getMessage());
        }
    }

    public void manualReconciliation(LocalDate date) {
        log.info("Manual reconciliation triggered for date: {}", date);
        dailyReconciliation();
    }
}