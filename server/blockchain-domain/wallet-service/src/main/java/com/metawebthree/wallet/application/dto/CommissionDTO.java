package com.metawebthree.wallet.application.dto;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(description = "Commission relationship information")
public class CommissionDTO {

    @Schema(description = "Account address")
    private String account;

    @Schema(description = "Upline address")
    private String upline;

    @Schema(description = "Level in referral tree")
    private Integer level;

    @Schema(description = "Downline count")
    private Integer downlineCount;

    public CommissionDTO() {}

    public CommissionDTO(String account, String upline, Integer level, Integer downlineCount) {
        this.account = account;
        this.upline = upline;
        this.level = level;
        this.downlineCount = downlineCount;
    }

    public String getAccount() { return account; }
    public void setAccount(String account) { this.account = account; }
    public String getUpline() { return upline; }
    public void setUpline(String upline) { this.upline = upline; }
    public Integer getLevel() { return level; }
    public void setLevel(Integer level) { this.level = level; }
    public Integer getDownlineCount() { return downlineCount; }
    public void setDownlineCount(Integer downlineCount) { this.downlineCount = downlineCount; }
}
