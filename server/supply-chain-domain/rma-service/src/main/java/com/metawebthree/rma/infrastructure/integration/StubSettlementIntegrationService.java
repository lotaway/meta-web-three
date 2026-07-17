package com.metawebthree.rma.infrastructure.integration;

import com.metawebthree.rma.application.integration.SettlementIntegrationService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;

/**
 * Stub implementation of SettlementIntegrationService.
 * Logs integration points. Replace with real RPC/HTTP client when settlement service is available.
 */
@Slf4j
@Component
@ConditionalOnMissingBean(SettlementIntegrationService.class)
public class StubSettlementIntegrationService implements SettlementIntegrationService {

    @Override
    public void triggerRefund(Long rmaId, String rmaNo, String orderNo, Long customerId, BigDecimal refundAmount, String dispositionType) {
        log.info("[INTEGRATION-STUB] Settlement triggerRefund called — would create refund settlement");
        log.info("  rmaId={}, rmaNo={}, orderNo={}, customerId={}, refundAmount={}, dispositionType={}",
                rmaId, rmaNo, orderNo, customerId, refundAmount, dispositionType);
    }
}
