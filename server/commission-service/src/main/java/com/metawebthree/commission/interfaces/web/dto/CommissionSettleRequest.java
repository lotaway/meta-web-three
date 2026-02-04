package com.metawebthree.commission.interfaces.web.dto;

import java.time.LocalDateTime;

public class CommissionSettleRequest {
    private LocalDateTime executeBefore;

    public LocalDateTime getExecuteBefore() { return executeBefore; }
    public void setExecuteBefore(LocalDateTime executeBefore) { this.executeBefore = executeBefore; }
}
