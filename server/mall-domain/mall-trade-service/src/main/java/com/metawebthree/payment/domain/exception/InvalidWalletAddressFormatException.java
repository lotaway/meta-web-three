package com.metawebthree.payment.domain.exception;

public class InvalidWalletAddressFormatException extends RiskControlException {
    public InvalidWalletAddressFormatException() {
        super("Invalid wallet address format");
    }
}