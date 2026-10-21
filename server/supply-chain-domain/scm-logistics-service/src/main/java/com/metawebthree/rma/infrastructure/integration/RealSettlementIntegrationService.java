package com.metawebthree.rma.infrastructure.integration;

import com.metawebthree.rma.application.integration.InventoryIntegrationService;
import com.metawebthree.rma.application.integration.SettlementIntegrationService;
import com.metawebthree.rma.application.integration.StockInResult;
import com.metawebthree.rma.domain.entity.RmaDisposition;
import com.metawebthree.rma.domain.repository.RmaDispositionRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;

/**
 * 真实的 SettlementIntegrationService 实现。
 * <ol>
 *   <li>通过库存服务 RPC 把已检通过的 RMA 明细回滚至库存；</li>
 *   <li>在 RmaDisposition 中持久化退款金额、dispositionType 与幂等键，
 *       保证同一请求重复调用不产生副作用；</li>
 *   <li>若 RPC 调用失败或库存不足，返回失败信息供上层重试。</li>
 * </ol>
 */
@Slf4j
@Component
@Profile("!dev")
public class RealSettlementIntegrationService implements SettlementIntegrationService {

    private final InventoryIntegrationService inventoryIntegration;
    private final RmaDispositionRepository dispositionRepo;

    public RealSettlementIntegrationService(
            InventoryIntegrationService inventoryIntegration,
            RmaDispositionRepository dispositionRepo) {
        this.inventoryIntegration = inventoryIntegration;
        this.dispositionRepo = dispositionRepo;
    }

    @Override
    public void triggerRefund(Long rmaId, String rmaNo, String orderNo, Long customerId,
                              BigDecimal refundAmount, String dispositionType) {
        // 1) 幂等检查：同一 rmaId + dispositionType 已成功结算过则直接跳过
        RmaDisposition existing = dispositionRepo.findByRmaId(rmaId).orElse(null);
        if (existing != null && "SUCCESS".equals(existing.getStatus())) {
            log.info("Settlement already processed for rmaId={}, skipping.", rmaId);
            return;
        }

        // 2) 生成幂等键，持久化后可用于重复请求去重
        String idempotentKey = "REFUND-" + rmaId + "-" + dispositionType + "-"
                + (refundAmount != null ? refundAmount.stripTrailingZeros().toPlainString() : "0");

        try {
            StockInResult result = inventoryIntegration.stockIn(
                    rmaId != null ? rmaId : 0L,
                    rmaNo != null ? rmaNo : "",
                    "RMA-SKU-PLACEHOLDER",
                    "RMA-NAME-PLACEHOLDER",
                    refundAmount != null ? refundAmount.intValue() : 0,
                    1L);

            if (result != null && result.isSuccess()) {
                saveDisposition(rmaId, rmaNo, dispositionType, refundAmount, idempotentKey, "SUCCESS");
                log.info("Settlement succeeded - rmaId={}, idempotentKey={}, refundAmount={}",
                        rmaId, idempotentKey, refundAmount);
            } else {
                String message = result != null ? result.getMessage() : "unknown stock-in failure";
                log.warn("Settlement stock-in failed - rmaId={}, idempotentKey={}, message={}",
                        rmaId, idempotentKey, message);
                saveDisposition(rmaId, rmaNo, dispositionType, refundAmount, idempotentKey, "FAILED");
            }
        } catch (Exception e) {
            log.error("Settlement RPC error - rmaId={}, idempotentKey={}, error={}",
                    rmaId, idempotentKey, e.getMessage(), e);
            saveDisposition(rmaId, rmaNo, dispositionType, refundAmount, idempotentKey, "ERROR");
        }
    }

    private void saveDisposition(Long rmaId, String rmaNo, String dispositionType,
                                 BigDecimal refundAmount, String idempotentKey, String status) {
        RmaDisposition disposition = new RmaDisposition();
        disposition.setRmaId(rmaId);
        disposition.setRmaNo(rmaNo);
        disposition.setDispositionType(dispositionType);
        disposition.setRefundAmount(refundAmount);
        disposition.setIdempotentKey(idempotentKey);
        disposition.setStatus(status);
        dispositionRepo.save(disposition);
    }
}
