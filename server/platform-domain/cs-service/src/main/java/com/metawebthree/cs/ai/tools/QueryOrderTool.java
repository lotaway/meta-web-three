package com.metawebthree.cs.ai.tools;

import com.metawebthree.common.generated.rpc.GetOrderByUserIdRequest;
import com.metawebthree.common.generated.rpc.GetOrderByUserIdResponse;
import com.metawebthree.common.generated.rpc.OrderService;
import com.metawebthree.cs.dto.AiToolRequest;
import com.metawebthree.cs.dto.AiToolResult;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import java.util.Map;

@Component
public class QueryOrderTool extends AiTool {
    @DubboReference(check = false, lazy = true)
    private OrderService orderService;

    public QueryOrderTool() {
        super("query_order", "根据订单ID查询订单状态、商品信息、金额等详细信息");
    }

    @Override
    public AiToolResult execute(AiToolRequest request) {
        Object orderIdObj = request.getParams().get("orderId");
        if (orderIdObj == null) {
            return AiToolResult.failure(getName(), "缺少参数 orderId");
        }
        try {
            Long orderId = Long.parseLong(orderIdObj.toString());
            GetOrderByUserIdRequest grpcRequest = GetOrderByUserIdRequest.newBuilder()
                    .setId(orderId)
                    .build();
            GetOrderByUserIdResponse response = orderService.getOrderByUserId(grpcRequest);
            
            if (response != null && response.getOrdersCount() > 0) {
                var order = response.getOrders(0);
                String json = String.format(
                    "{\"id\":%d,\"userId\":%d,\"orderNo\":\"%s\",\"orderStatus\":\"%s\",\"orderType\":\"%s\"}",
                    order.getId(),
                    order.getUserId(),
                    order.getOrderNo(),
                    order.getOrderStatus(),
                    order.getOrderType()
                );
                return AiToolResult.success(getName(), json);
            }
            return AiToolResult.failure(getName(), "未找到订单: " + orderId);
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
