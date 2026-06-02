package com.metawebthree.finance.application.command.arap.dto;

import java.math.BigDecimal;

public class ArReceiveCommand {
    private Long arId;
    private BigDecimal receiveAmount;
    private String paymentMethod;
    private String bankAccount;
    private String remark;
    private Long operatorId;
    private String operatorName;

    public Long getArId() { return arId; }
    public void setArId(Long arId) { this.arId = arId; }
    public BigDecimal getReceiveAmount() { return receiveAmount; }
    public void setReceiveAmount(BigDecimal receiveAmount) { this.receiveAmount = receiveAmount; }
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