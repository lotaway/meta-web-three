package com.metawebthree.cs.ai.tools;

import com.metawebthree.common.generated.rpc.CloseOrderRequest;
import com.metawebthree.common.generated.rpc.CloseOrderResponse;
import com.metawebthree.common.generated.rpc.OrderService;
import com.metawebthree.cs.dto.AiToolRequest;
import com.metawebthree.cs.dto.AiToolResult;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import java.util.Map;

@Component
public class CancelOrderTool extends AiTool {
    @DubboReference(check = false, lazy = true)
    private OrderService orderService;

    public CancelOrderTool() {
        super("cancel_order", "取消指定订单，需要提供订单ID和取消原因");
    }

    @Override
    public AiToolResult execute(AiToolRequest request) {
        Map<String, Object> params = request.getParams();
        Object orderIdObj = params.get("orderId");
        Object reason = params.get("reason");
        if (orderIdObj == null || reason == null) {
            return AiToolResult.failure(getName(), "缺少参数 orderId 或 reason");
        }
        try {
            String orderIds = orderIdObj.toString();
            CloseOrderRequest grpcRequest = CloseOrderRequest.newBuilder()
                    .setIds(orderIds)
                    .setNote(reason.toString())
                    .build();
            CloseOrderResponse response = orderService.closeOrder(grpcRequest);
            
            if (response != null && response.getSuccess()) {
                return AiToolResult.success(getName(), "订单取消成功");
            }
            String errorMsg = response != null ? response.getMessage() : "未知错误";
            return AiToolResult.failure(getName(), "取消订单失败: " + errorMsg);
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
