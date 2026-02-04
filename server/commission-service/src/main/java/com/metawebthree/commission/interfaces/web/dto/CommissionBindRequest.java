package com.metawebthree.commission.interfaces.web.dto;

public class CommissionBindRequest {
    private Long userId;
    private Long parentUserId;

    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }
    public Long getParentUserId() { return parentUserId; }
    public void setParentUserId(Long parentUserId) { this.parentUserId = parentUserId; }
}
