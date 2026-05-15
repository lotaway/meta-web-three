package com.metawebthree.cs.ai.tools;

import com.metawebthree.cs.dto.AiToolRequest;
import com.metawebthree.cs.dto.AiToolResult;
import org.springframework.web.client.RestTemplate;

import java.util.Map;

public class InitiateRefundTool extends AiTool {
    private static final String SERVICE = "/order-service/returnApply";
    private final RestTemplate restTemplate;
    private final String gatewayUrl;

    public InitiateRefundTool(RestTemplate restTemplate, String gatewayUrl) {
        super("initiate_refund", "为指定订单发起退货退款申请，需要提供订单ID和退款原因");
        this.restTemplate = restTemplate;
        this.gatewayUrl = gatewayUrl;
    }

    @Override
    public AiToolResult execute(AiToolRequest request) {
        Map<String, Object> params = request.getParams();
        Object orderId = params.get("orderId");
        Object reason = params.get("reason");
        if (orderId == null || reason == null) {
            return AiToolResult.failure(getName(), "缺少参数 orderId 或 reason");
        }
        try {
            String json = restTemplate.postForObject(
                    gatewayUrl + SERVICE + "/create",
                    Map.of("orderId", orderId, "reason", reason, "userId", request.getUserId()),
                    String.class);
            return AiToolResult.success(getName(), json);
        } catch (Exception e) {
            return AiToolResult.failure(getName(), "发起退款失败: " + e.getMessage());
        }
    }

    @Override
    public Map<String, Object> getParameterSchema() {
        return Map.of(
                "type", "object",
                "properties", Map.of(
                        "orderId", Map.of("type", "string", "description", "订单ID"),
                        "reason", Map.of("type", "string", "description", "退款原因说明")
                ),
                "required", java.util.List.of("orderId", "reason")
        );
    }
}
