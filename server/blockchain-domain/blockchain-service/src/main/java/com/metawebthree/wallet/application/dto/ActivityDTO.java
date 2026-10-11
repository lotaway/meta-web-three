package com.metawebthree.wallet.application.dto;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(description = "Activity information")
public class ActivityDTO {

    @Schema(description = "Activity PDA address")
    private String activityAddress;

    @Schema(description = "Authority wallet address")
    private String authority;

    @Schema(description = "Start timestamp")
    private Long startTime;

    @Schema(description = "End timestamp")
    private Long endTime;

    @Schema(description = "Entry fee")
    private Long entryFee;

    @Schema(description = "Reward percentages")
    private Integer[] rewardPercentages;

    @Schema(description = "Total pool amount")
    private Long totalPool;

    @Schema(description = "Participant count")
    private Long participantCount;

    @Schema(description = "Transaction signature")
    private String txSignature;

    public ActivityDTO() {}

    public ActivityDTO(String activityAddress, String authority, Long startTime, Long endTime,
                       Long entryFee, Integer[] rewardPercentages, Long totalPool,
                       Long participantCount, String txSignature) {
        this.activityAddress = activityAddress;
        this.authority = authority;
        this.startTime = startTime;
        this.endTime = endTime;
        this.entryFee = entryFee;
        this.rewardPercentages = rewardPercentages;
        this.totalPool = totalPool;
        this.participantCount = participantCount;
        this.txSignature = txSignature;
    }

    public String getActivityAddress() { return activityAddress; }
    public void setActivityAddress(String activityAddress) { this.activityAddress = activityAddress; }
    public String getAuthority() { return authority; }
    public void setAuthority(String authority) { this.authority = authority; }
    public Long getStartTime() { return startTime; }
    public void setStartTime(Long startTime) { this.startTime = startTime; }
    public Long getEndTime() { return endTime; }
    public void setEndTime(Long endTime) { this.endTime = endTime; }
    public Long getEntryFee() { return entryFee; }
    public void setEntryFee(Long entryFee) { this.entryFee = entryFee; }
    public Integer[] getRewardPercentages() { return rewardPercentages; }
    public void setRewardPercentages(Integer[] rewardPercentages) { this.rewardPercentages = rewardPercentages; }
    public Long getTotalPool() { return totalPool; }
    public void setTotalPool(Long totalPool) { this.totalPool = totalPool; }
    public Long getParticipantCount() { return participantCount; }
    public void setParticipantCount(Long participantCount) { this.participantCount = participantCount; }
    public String getTxSignature() { return txSignature; }
    public void setTxSignature(String txSignature) { this.txSignature = txSignature; }
}
