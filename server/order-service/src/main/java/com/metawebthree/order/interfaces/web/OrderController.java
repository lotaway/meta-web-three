package com.metawebthree.order.interfaces.web;

import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.common.constants.RequestHeaderKeys;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.order.application.OrderApplicationService;
import com.metawebthree.order.application.OrderApplicationService.OrderCreateRequest;
import com.metawebthree.order.application.OrderApplicationService.OrderWithItems;

import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;

@Slf4j
@RestController
@RequestMapping("/order")
public class OrderController {

    @Autowired
    private OrderApplicationService orderService;

    @PostMapping("/create")
    public ApiResponse<Long> create(@RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestBody OrderCreateRequest request) {
        Long orderId = orderService.createOrder(userId, request.getRemark(), request.getItems());
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

    @PostMapping("/refund/{id}")
    public ApiResponse<?> refund(@RequestHeader(HeaderConstants.USER_ID) Long userId,
            @PathVariable("id") Long id) {
        boolean refunded = orderService.refundOrder(id, userId);
        if (refunded) {
            return ApiResponse.success();
        }
        return ApiResponse.error(ResponseStatus.ORDER_STATUS_INVALID);
    }

    @GetMapping("/micro-service-test")
    public String microServiceTest() {
        return "Order service is running";
    }
}
