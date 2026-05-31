package com.metawebthree.payment.domain.exception;

public class BlacklistedWalletAddressException extends RiskControlException {
    public BlacklistedWalletAddressException() {
        super("Wallet address is blacklisted");
    }
}