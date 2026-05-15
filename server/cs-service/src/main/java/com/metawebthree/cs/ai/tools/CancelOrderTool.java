package com.metawebthree.cs.ai.tools;

import com.metawebthree.cs.dto.AiToolRequest;
import com.metawebthree.cs.dto.AiToolResult;
import org.springframework.web.client.RestTemplate;

import java.util.Map;

public class CancelOrderTool extends AiTool {
    private static final String SERVICE = "/order-service/order";
    private final RestTemplate restTemplate;
    private final String gatewayUrl;

    public CancelOrderTool(RestTemplate restTemplate, String gatewayUrl) {
        super("cancel_order", "取消指定订单，需要提供订单ID和取消原因");
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
                    gatewayUrl + SERVICE + "/update/close",
                    Map.of("ids", String.valueOf(orderId), "note", reason),
                    String.class);
            return AiToolResult.success(getName(), json);
        } catch (Exception e) {
            return AiToolResult.failure(getName(), "取消订单失败: " + e.getMessage());
        }
    }

    @Override
    public Map<String, Object> getParameterSchema() {
        return Map.of(
                "type", "object",
                "properties", Map.of(
                        "orderId", Map.of("type", "string", "description", "订单ID"),
                        "reason", Map.of("type", "string", "description", "取消原因")
                ),
                "required", java.util.List.of("orderId", "reason")
        );
    }
}
