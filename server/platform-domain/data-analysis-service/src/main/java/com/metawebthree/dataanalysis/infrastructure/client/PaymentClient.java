package com.metawebthree.dataanalysis.infrastructure.client;

import com.metawebthree.common.generated.rpc.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

@Slf4j
@Component
public class PaymentClient {

    @DubboReference(check = false, lazy = true)
    private PaymentService paymentService;

    /**
     * Get pending payments count
     * Primary source is order-service via OrderClient.getPendingPaymentsCount()
     * This provides payment-specific pending count if available
     * @return count of pending payments
     */
    public Long getPendingPaymentsCount() {
        // Primary source is order-service
        // This is a fallback for payment-specific data
        try {
            GetPaymentStatisticsResponse response = paymentService.getPaymentStatistics(
                    GetPaymentStatisticsRequest.getDefaultInstance()
            );
            return response.getStatistics().getPendingPayments();
        } catch (Exception e) {
            log.error("Failed to get pending payments count via Dubbo", e);
            return 0L;
        }
    }

    /**
     * Get payment statistics
     * @return map of payment statistics
     */
    public Map<String, Long> getPaymentStatistics() {
        try {
            GetPaymentStatisticsResponse response = paymentService.getPaymentStatistics(
                    GetPaymentStatisticsRequest.getDefaultInstance()
            );
            Map<String, Long> stats = new HashMap<>();
            PaymentStatistics statistics = response.getStatistics();
            stats.put("totalPayments", statistics.getTotalPayments());
            stats.put("successPayments", statistics.getSuccessPayments());
            stats.put("failedPayments", statistics.getFailedPayments());
            stats.put("pendingPayments", statistics.getPendingPayments());
            return stats;
        } catch (Exception e) {
            log.error("Failed to get payment statistics via Dubbo", e);
            return new HashMap<>();
        }
    }
}