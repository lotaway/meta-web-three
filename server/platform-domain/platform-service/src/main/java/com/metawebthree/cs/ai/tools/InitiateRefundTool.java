package com.metawebthree.cs.ai.tools;

import com.metawebthree.common.generated.rpc.CreateReturnApplyRequest;
import com.metawebthree.common.generated.rpc.CreateReturnApplyResponse;
import com.metawebthree.common.generated.rpc.OrderService;
import com.metawebthree.cs.dto.AiToolRequest;
import com.metawebthree.cs.dto.AiToolResult;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import java.util.Map;

@Component
public class InitiateRefundTool extends AiTool {
    @DubboReference(check = false, lazy = true)
    private OrderService orderService;

    public InitiateRefundTool() {
        super("initiate_refund", "为指定订单发起退货退款申请，需要提供订单ID和退款原因");
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
            Long orderId = Long.parseLong(orderIdObj.toString());
            Long userId = request.getUserId();
            
            CreateReturnApplyRequest grpcRequest = CreateReturnApplyRequest.newBuilder()
                    .setOrderId(orderId)
                    .setReason(reason.toString())
                    .setUserId(userId != null ? userId : 0L)
                    .build();
            CreateReturnApplyResponse response = orderService.createReturnApply(grpcRequest);
            
            if (response != null && response.getSuccess()) {
                return AiToolResult.success(getName(), "退款申请已发起");
            }
            String errorMsg = response != null ? response.getMessage() : "未知错误";
            return AiToolResult.failure(getName(), "发起退款失败: " + errorMsg);
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
