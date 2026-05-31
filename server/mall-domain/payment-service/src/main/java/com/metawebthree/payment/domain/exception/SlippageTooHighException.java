package com.metawebthree.payment.domain.exception;

public class SlippageTooHighException extends RiskControlException {
    public SlippageTooHighException(String expectedRate, String actualRate, String slippage) {
        super("Slippage too high. Expected: " + expectedRate + ", Actual: " + actualRate + ", Slippage: " + slippage + "%");
    }
}