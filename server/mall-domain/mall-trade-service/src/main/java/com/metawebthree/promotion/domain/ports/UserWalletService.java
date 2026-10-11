package com.metawebthree.promotion.domain.ports;

public interface UserWalletService {
    /**
     * Map a userId to their current primary wallet address.
     * @param userId Internal userID.
     * @return Hex string wallet address (including 0x).
     */
    String getWalletAddressByUserId(Long userId);
}
