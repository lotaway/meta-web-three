package com.metawebthree.cs.ai.tools;

import com.metawebthree.common.generated.rpc.OrderService;
import com.metawebthree.common.generated.rpc.QueryLogisticsRequest;
import com.metawebthree.common.generated.rpc.QueryLogisticsResponse;
import com.metawebthree.cs.dto.AiToolRequest;
import com.metawebthree.cs.dto.AiToolResult;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import java.util.Map;

@Component
public class QueryLogisticsTool extends AiTool {
    @DubboReference(check = false, lazy = true)
    private OrderService orderService;

    public QueryLogisticsTool() {
        super("query_logistics", "根据订单ID查询物流信息，包括快递公司、运单号、物流轨迹");
    }

    @Override
    public AiToolResult execute(AiToolRequest request) {
        Object orderIdObj = request.getParams().get("orderId");
        if (orderIdObj == null) {
            return AiToolResult.failure(getName(), "缺少参数 orderId");
        }
        try {
            Long orderId = Long.parseLong(orderIdObj.toString());
            QueryLogisticsRequest grpcRequest = QueryLogisticsRequest.newBuilder()
                    .setOrderId(orderId)
                    .build();
            QueryLogisticsResponse response = orderService.queryLogistics(grpcRequest);
            
            if (response != null) {
                String json = String.format(
                    "{\"logisticsCompany\":\"%s\",\"trackingNumber\":\"%s\",\"traces\":%s}",
                    response.getLogisticsCompany(),
                    response.getTrackingNumber(),
                    response.getTracesList()
                );
                return AiToolResult.success(getName(), json);
            }
            return AiToolResult.failure(getName(), "未找到物流信息");
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
