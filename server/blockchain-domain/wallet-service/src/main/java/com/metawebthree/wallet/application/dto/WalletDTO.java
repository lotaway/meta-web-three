package com.metawebthree.wallet.application.dto;

import java.math.BigDecimal;

public class WalletDTO {
    private Long id;
    private String userId;
    private String chainType;
    private String address;
    private BigDecimal balance;
    private String status;

    public WalletDTO() {}

    public WalletDTO(Long id, String userId, String chainType, String address, BigDecimal balance, String status) {
        this.id = id;
        this.userId = userId;
        this.chainType = chainType;
        this.address = address;
        this.balance = balance;
        this.status = status;
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getUserId() { return userId; }
    public void setUserId(String userId) { this.userId = userId; }
    public String getChainType() { return chainType; }
    public void setChainType(String chainType) { this.chainType = chainType; }
    public String getAddress() { return address; }
    public void setAddress(String address) { this.address = address; }
    public BigDecimal getBalance() { return balance; }
    public void setBalance(BigDecimal balance) { this.balance = balance; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
}