package com.metawebthree.wallet.application.dto;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(description = "Request to create an activity")
public class CreateActivityRequest {

    @Schema(description = "Activity authority wallet address")
    private String authority;

    @Schema(description = "Start timestamp (Unix seconds)")
    private Long startTime;

    @Schema(description = "End timestamp (Unix seconds)")
    private Long endTime;

    @Schema(description = "Entry fee in tokens")
    private Long entryFee;

    @Schema(description = "Reward percentages [1st, 2nd, 3rd] * 100 (e.g. 5000 = 50%)")
    private Integer[] rewardPercentages;

    @Schema(description = "Payment token mint address")
    private String paymentMint;

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
    public String getPaymentMint() { return paymentMint; }
    public void setPaymentMint(String paymentMint) { this.paymentMint = paymentMint; }
}
