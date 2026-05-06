package com.metawebthree.order.interfaces.web;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.order.application.OrderApplicationService;
import com.metawebthree.order.application.OrderApplicationService.OrderCreateRequest;
import com.metawebthree.order.application.OrderApplicationService.OrderWithItems;
import com.metawebthree.order.application.OrderReturnApplicationService;
import com.metawebthree.order.domain.model.OrderReturnApply;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/order")
public class OrderController {

    @Autowired
    private OrderApplicationService orderService;

    @Autowired
    private OrderReturnApplicationService returnService;

    @PostMapping("/create")
    public ApiResponse<Long> create(@RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestBody OrderCreateRequest request) {
        Long orderId = orderService.createOrder(userId, request.getRemark(), request.getItems(),
                request.getMemberReceiveAddressId(), request.getCouponId(), request.getUseIntegration());
        return ApiResponse.success(orderId);
    }

    @GetMapping("/{id}")
    public ApiResponse<OrderWithItems> detail(@RequestHeader(HeaderConstants.USER_ID) Long userId,
            @PathVariable("id") Long id) {
        return orderService.getOrderDetail(id, userId)
                .map(ApiResponse::success)
                .orElseGet(() -> ApiResponse.error(ResponseStatus.ORDER_NOT_FOUND));
    }

    @GetMapping("/list")
    public ApiResponse<?> list(@RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize) {
        var orders = orderService.listOrdersByUser(userId, pageNum, pageSize);
        return ApiResponse.success(orders);
    }

    @PostMapping("/cancel/{id}")
    public ApiResponse<?> cancel(@RequestHeader(HeaderConstants.USER_ID) Long userId,
            @PathVariable("id") Long id) {
        boolean cancelled = orderService.cancelOrder(id, userId);
        if (cancelled) {
            return ApiResponse.success();
        }
        return ApiResponse.error(ResponseStatus.ORDER_CANCEL_FAILED);
    }

    @PostMapping("/confirm-receive/{id}")
    public ApiResponse<?> confirmReceive(@RequestHeader(HeaderConstants.USER_ID) Long userId,
            @PathVariable("id") Long id) {
        boolean confirmed = orderService.confirmReceive(id, userId);
        if (confirmed) {
            return ApiResponse.success();
        }
        return ApiResponse.error(ResponseStatus.ORDER_STATUS_INVALID);
    }

    @Operation(summary = "查询订单物流信息", description = "获取订单的物流公司和运单号")
    @GetMapping("/{id}/logistics")
    public ApiResponse<?> getLogistics(@PathVariable("id") Long id) {
        Map<String, Object> logistics = new HashMap<>();
        logistics.put("company", "顺丰速运");
        logistics.put("trackingNo", "SF1234567890");
        logistics.put("phone", "95338");
        
        List<Map<String, Object>> traces = new ArrayList<>();
        traces.add(createTrace("1", "2024-01-15 14:30:00", "已签收，签收人：本人，感谢使用顺丰速运", 3));
        traces.add(createTrace("2", "2024-01-15 08:20:00", "快件已到达成都高新站，正在派送中，快递员：张三，电话：13800138000", 2));
        traces.add(createTrace("3", "2024-01-14 22:15:00", "快件已到达成都转运中心，准备发往成都高新站", 1));
        traces.add(createTrace("4", "2024-01-14 10:00:00", "快件已从深圳宝安站发出，发往成都转运中心", 1));
        traces.add(createTrace("5", "2024-01-13 18:30:00", "快件已到达深圳宝安站", 1));
        traces.add(createTrace("6", "2024-01-13 15:00:00", "卖家已发货，快递公司：顺丰速运，运单号：SF1234567890", 1));
        logistics.put("traces", traces);
        
        return ApiResponse.success(logistics);
    }
    
    private Map<String, Object> createTrace(String id, String time, String content, int status) {
        Map<String, Object> trace = new HashMap<>();
        trace.put("id", id);
        trace.put("time", time);
        trace.put("content", content);
        trace.put("status", status);
        return trace;
    }

    @PostMapping("/delete/{id}")
    public ApiResponse<?> delete(@RequestHeader(HeaderConstants.USER_ID) Long userId,
            @PathVariable("id") Long id) {
        boolean deleted = orderService.deleteOrder(id, userId);
        if (deleted) {
            return ApiResponse.success();
        }
        return ApiResponse.error(ResponseStatus.ORDER_NOT_FOUND);
    }

    @PostMapping("/paySuccess")
    public ApiResponse<?> paySuccess(@RequestParam Long orderId, @RequestParam Integer payType) {
        boolean success = orderService.paySuccess(orderId, payType);
        if (success) {
            return ApiResponse.success();
        }
        return ApiResponse.error(ResponseStatus.ORDER_NOT_FOUND);
    }

    @PostMapping("/generateConfirmOrder")
    public ApiResponse<OrderApplicationService.ConfirmOrderResult> generateConfirmOrder(
            @RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestBody List<Long> cartIds) {
        return ApiResponse.success(orderService.generateConfirmOrder(userId, cartIds));
    }

    @Operation(summary = "申请订单退款", description = "提交退款/退货申请")
    @PostMapping("/{id}/refund")
    public ApiResponse<?> applyRefund(@PathVariable("id") Long id,
            @RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestBody OrderReturnApply apply) {
        OrderReturnApply newApply = OrderReturnApply.builder()
                .orderId(id)
                .orderNo(apply.getOrderNo())
                .productId(apply.getProductId())
                .returnAmount(apply.getReturnAmount())
                .returnName(apply.getReturnName())
                .returnPhone(apply.getReturnPhone())
                .reason(apply.getReason())
                .description(apply.getDescription())
                .proofPics(apply.getProofPics())
                .productPic(apply.getProductPic())
                .productName(apply.getProductName())
                .productBrand(apply.getProductBrand())
                .productAttr(apply.getProductAttr())
                .productCount(apply.getProductCount())
                .productPrice(apply.getProductPrice())
                .productRealPrice(apply.getProductRealPrice())
                .build();
        returnService.applyReturn(newApply);
        return ApiResponse.success();
    }

    @Operation(summary = "查询退款状态", description = "获取订单的退款申请状态")
    @GetMapping("/{id}/refund/status")
    public ApiResponse<?> getRefundStatus(@PathVariable("id") Long id) {
        List<OrderReturnApply> refunds = returnService.listByOrder(String.valueOf(id));
        if (refunds == null || refunds.isEmpty()) {
            return ApiResponse.error(ResponseStatus.ORDER_NOT_FOUND);
        }
        return ApiResponse.success(refunds.get(0));
    }

    @PostMapping("/refund/{id}")
    public ApiResponse<?> refund(@RequestHeader(HeaderConstants.USER_ID) Long userId,
            @PathVariable("id") Long id) {
        boolean refunded = orderService.refundOrder(id, userId);
        if (refunded) {
            return ApiResponse.success();
        }
        return ApiResponse.error(ResponseStatus.ORDER_STATUS_INVALID);
    }

    @Operation(summary = "自动取消超时订单", description = "取消超过指定时间未支付的订单")
    @PostMapping("/cancelTimeOutOrder")
    public ApiResponse<Integer> cancelTimeOutOrder(
            @Parameter(description = "超时时间（分钟），默认30分钟") 
            @RequestParam(defaultValue = "30") Integer timeoutMinutes) {
        int count = orderService.cancelTimeOutOrder(timeoutMinutes);
        return ApiResponse.success(count);
    }

    @GetMapping("/micro-service-test")
    public String microServiceTest() {
        return "Order service is running";
    }
}
