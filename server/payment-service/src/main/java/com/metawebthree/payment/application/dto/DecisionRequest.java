package com.metawebthree.payment.application.dto;

import java.util.Map;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class DecisionRequest {
    private String bizOrderId;
    private Long userId;
    private String deviceId;
    private String scene;
    private Map<String, Object> context;
}
