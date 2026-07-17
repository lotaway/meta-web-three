package com.metawebthree.rma.application.integration;

import java.math.BigDecimal;

/**
 * Integration interface for settlement service.
 * Implementations handle financial settlement operations when RMA disposition is executed.
 */
public interface SettlementIntegrationService {

    /**
     * Trigger a refund settlement for the disposition type.
     *
     * @param rmaId           RMA order ID
     * @param rmaNo           RMA order number
     * @param orderNo         original order number
     * @param customerId      customer receiving the refund
     * @param refundAmount    refund amount
     * @param dispositionType disposition type (REFUND, REPLACEMENT, etc.)
     */
    void triggerRefund(Long rmaId, String rmaNo, String orderNo, Long customerId, BigDecimal refundAmount, String dispositionType);
}
