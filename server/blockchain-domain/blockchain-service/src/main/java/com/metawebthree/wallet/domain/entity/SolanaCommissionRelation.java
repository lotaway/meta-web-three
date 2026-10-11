package com.metawebthree.wallet.domain.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;

@TableName("tb_solana_commission_relation")
public class SolanaCommissionRelation {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String accountAddress;
    private String uplineAddress;
    private Integer level;
    private Integer downlineCount;
    private String txSignature;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public SolanaCommissionRelation() {}

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getAccountAddress() { return accountAddress; }
    public void setAccountAddress(String accountAddress) { this.accountAddress = accountAddress; }
    public String getUplineAddress() { return uplineAddress; }
    public void setUplineAddress(String uplineAddress) { this.uplineAddress = uplineAddress; }
    public Integer getLevel() { return level; }
    public void setLevel(Integer level) { this.level = level; }
    public Integer getDownlineCount() { return downlineCount; }
    public void setDownlineCount(Integer downlineCount) { this.downlineCount = downlineCount; }
    public String getTxSignature() { return txSignature; }
    public void setTxSignature(String txSignature) { this.txSignature = txSignature; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}
