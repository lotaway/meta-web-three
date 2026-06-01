package com.metawebthree.supplier.infrastructure.rpc;

import org.springframework.stereotype.Component;

@Component
public class SettlementServiceClient {

    public void createSettlement(String orderId, String supplierCode, Double amount) {
        // Placeholder for settlement service integration
        // Will be implemented when settlement service RPC is available
    }

    public void updateSettlementStatus(String settlementId, String status) {
        // Placeholder for settlement status update
    }
}