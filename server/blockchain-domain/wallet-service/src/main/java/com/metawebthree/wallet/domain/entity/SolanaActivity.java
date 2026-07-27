package com.metawebthree.wallet.domain.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;

@TableName("tb_solana_activity")
public class SolanaActivity {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String activityAddress;
    private String authorityAddress;
    private Long startTime;
    private Long endTime;
    private Long entryFee;
    private String rewardPcts;
    private String paymentMint;
    private Long totalPool;
    private Integer participantCount;
    private String txSignature;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public SolanaActivity() {}

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getActivityAddress() { return activityAddress; }
    public void setActivityAddress(String activityAddress) { this.activityAddress = activityAddress; }
    public String getAuthorityAddress() { return authorityAddress; }
    public void setAuthorityAddress(String authorityAddress) { this.authorityAddress = authorityAddress; }
    public Long getStartTime() { return startTime; }
    public void setStartTime(Long startTime) { this.startTime = startTime; }
    public Long getEndTime() { return endTime; }
    public void setEndTime(Long endTime) { this.endTime = endTime; }
    public Long getEntryFee() { return entryFee; }
    public void setEntryFee(Long entryFee) { this.entryFee = entryFee; }
    public String getRewardPcts() { return rewardPcts; }
    public void setRewardPcts(String rewardPcts) { this.rewardPcts = rewardPcts; }
    public String getPaymentMint() { return paymentMint; }
    public void setPaymentMint(String paymentMint) { this.paymentMint = paymentMint; }
    public Long getTotalPool() { return totalPool; }
    public void setTotalPool(Long totalPool) { this.totalPool = totalPool; }
    public Integer getParticipantCount() { return participantCount; }
    public void setParticipantCount(Integer participantCount) { this.participantCount = participantCount; }
    public String getTxSignature() { return txSignature; }
    public void setTxSignature(String txSignature) { this.txSignature = txSignature; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}
