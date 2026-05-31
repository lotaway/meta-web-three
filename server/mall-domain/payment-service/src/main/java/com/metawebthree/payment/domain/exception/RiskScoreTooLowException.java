package com.metawebthree.payment.domain.exception;

public class RiskScoreTooLowException extends RiskControlException {
    public RiskScoreTooLowException(int score, int requiredScore, String decision) {
        super("Risk score too low: " + score + " (Required: " + requiredScore + "). Decision: " + decision);
    }
}