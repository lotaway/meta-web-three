package com.metawebthree.payment.domain.exception;

public class TransactionFrequencyTooHighException extends RiskControlException {
    public TransactionFrequencyTooHighException(int limit, int current) {
        super("Transaction frequency too high. Hourly limit: " + limit + ", Current: " + current);
    }
}