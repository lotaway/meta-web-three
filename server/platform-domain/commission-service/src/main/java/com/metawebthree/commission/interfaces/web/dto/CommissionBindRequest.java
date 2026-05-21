package com.metawebthree.commission.interfaces.web.dto;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(description = "佣金绑定请求")
public class CommissionBindRequest {
    @Schema(description = "用户ID")
    private Long userId;
    @Schema(description = "上级用户ID")
    private Long parentUserId;

    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }
    public Long getParentUserId() { return parentUserId; }
    public void setParentUserId(Long parentUserId) { this.parentUserId = parentUserId; }
}
