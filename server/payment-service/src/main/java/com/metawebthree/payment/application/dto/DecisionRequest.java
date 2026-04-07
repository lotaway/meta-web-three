package com.metawebthree.payment.application.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import java.util.Map;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
@Schema(description = "风控决策请求")
public class DecisionRequest {
    @Schema(description = "业务订单ID")
    private String bizOrderId;
    @Schema(description = "用户ID")
    private Long userId;
    @Schema(description = "设备ID")
    private String deviceId;
    @Schema(description = "场景")
    private String scene;
    @Schema(description = "上下文信息")
    private Map<String, Object> context;
}
