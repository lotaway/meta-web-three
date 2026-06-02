package com.metawebthree.finance.application.command.arap.dto;

import java.math.BigDecimal;

public class ApPayCommand {
    private Long apId;
    private BigDecimal payAmount;
    private String paymentMethod;
    private String bankAccount;
    private String remark;
    private Long operatorId;
    private String operatorName;

    public Long getApId() { return apId; }
    public void setApId(Long apId) { this.apId = apId; }
    public BigDecimal getPayAmount() { return payAmount; }
    public void setPayAmount(BigDecimal payAmount) { this.payAmount = payAmount; }
    public String getPaymentMethod() { return paymentMethod; }
    public void setPaymentMethod(String paymentMethod) { this.paymentMethod = paymentMethod; }
    public String getBankAccount() { return bankAccount; }
    public void setBankAccount(String bankAccount) { this.bankAccount = bankAccount; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
    public Long getOperatorId() { return operatorId; }
    public void setOperatorId(Long operatorId) { this.operatorId = operatorId; }
    public String getOperatorName() { return operatorName; }
    public void setOperatorName(String operatorName) { this.operatorName = operatorName; }
}