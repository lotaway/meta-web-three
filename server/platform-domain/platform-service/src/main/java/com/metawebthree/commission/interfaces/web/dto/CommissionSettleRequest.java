package com.metawebthree.commission.interfaces.web.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import java.time.LocalDateTime;

@Schema(description = "佣金结算请求")
public class CommissionSettleRequest {
    @Schema(description = "执行时间")
    private LocalDateTime executeBefore;

    public LocalDateTime getExecuteBefore() { return executeBefore; }
    public void setExecuteBefore(LocalDateTime executeBefore) { this.executeBefore = executeBefore; }
}
