package com.metawebthree.promotion.domain.ports;

import java.math.BigDecimal;

public interface BlockchainService {
    /**
     * Publishes the Merkle Root for a coupon batch to the blockchain.
     * @param batchId The unique identifier for the batch.
     * @param merkleRoot The computed Merkle Root (hex string).
     */
    void setCouponBatchRoot(String batchId, String merkleRoot);

    /**
     * Creates a new wallet and returns the address.
     * @return The generated wallet address.
     */
    String createWallet();

    /**
     * Gets the balance of a wallet.
     * @param address The wallet address.
     * @return The balance.
     */
    BigDecimal getBalance(String address);

    /**
     * Transfers an amount from one wallet to another.
     * @param from The source wallet address.
     * @param to The destination wallet address.
     * @param amount The amount to transfer.
     * @return true if successful.
     */
    boolean transfer(String from, String to, BigDecimal amount);
}
