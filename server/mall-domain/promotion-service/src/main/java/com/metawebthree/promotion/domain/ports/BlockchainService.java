package com.metawebthree.promotion.domain.ports;

public interface BlockchainService {
    /**
     * Publishes the Merkle Root for a coupon batch to the blockchain.
     * @param batchId The unique identifier for the batch.
     * @param merkleRoot The computed Merkle Root (hex string).
     */
    void setCouponBatchRoot(String batchId, String merkleRoot);
}
