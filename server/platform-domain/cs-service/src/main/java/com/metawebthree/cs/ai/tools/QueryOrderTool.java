package com.metawebthree.cs.ai.tools;

import com.metawebthree.cs.dto.AiToolRequest;
import com.metawebthree.cs.dto.AiToolResult;
import org.springframework.web.client.RestTemplate;

import java.util.Map;

public class QueryOrderTool extends AiTool {
    private static final String SERVICE = "/order-service/order";
    private final RestTemplate restTemplate;
    private final String gatewayUrl;

    public QueryOrderTool(RestTemplate restTemplate, String gatewayUrl) {
        super("query_order", "根据订单ID查询订单状态、商品信息、金额等详细信息");
        this.restTemplate = restTemplate;
        this.gatewayUrl = gatewayUrl;
    }

    @Override
    public AiToolResult execute(AiToolRequest request) {
        Object orderId = request.getParams().get("orderId");
        if (orderId == null) {
            return AiToolResult.failure(getName(), "缺少参数 orderId");
        }
        try {
            String json = restTemplate.getForObject(
                    gatewayUrl + SERVICE + "/detail?orderId={id}", String.class, orderId);
            return AiToolResult.success(getName(), json);
        } catch (Exception e) {
            return AiToolResult.failure(getName(), "查询订单失败: " + e.getMessage());
        }
    }

    @Override
    public Map<String, Object> getParameterSchema() {
        return Map.of(
                "type", "object",
                "properties", Map.of(
                        "orderId", Map.of("type", "string", "description", "订单ID")
                ),
                "required", java.util.List.of("orderId")
        );
    }
}
