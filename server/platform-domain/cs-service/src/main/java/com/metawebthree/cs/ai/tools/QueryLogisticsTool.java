package com.metawebthree.cs.ai.tools;

import com.metawebthree.cs.dto.AiToolRequest;
import com.metawebthree.cs.dto.AiToolResult;
import org.springframework.web.client.RestTemplate;

import java.util.Map;

public class QueryLogisticsTool extends AiTool {
    private static final String SERVICE = "/order-service/order";
    private final RestTemplate restTemplate;
    private final String gatewayUrl;

    public QueryLogisticsTool(RestTemplate restTemplate, String gatewayUrl) {
        super("query_logistics", "根据订单ID查询物流信息，包括快递公司、运单号、物流轨迹");
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
            return AiToolResult.failure(getName(), "查询物流失败: " + e.getMessage());
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
