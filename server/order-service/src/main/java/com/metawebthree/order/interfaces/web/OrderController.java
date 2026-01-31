package com.metawebthree.order.interfaces.web;

import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.order.application.OrderApplicationService;
import com.metawebthree.order.application.OrderApplicationService.OrderCreateRequest;
import com.metawebthree.order.application.OrderApplicationService.OrderWithItems;
import com.metawebthree.order.domain.model.OrderDO;

import org.springframework.beans.factory.annotation.Autowired;

@Slf4j
@RestController
@RequestMapping("/order")
public class OrderController {

    @Autowired
    private OrderApplicationService orderService;

    @PostMapping("/create")
    public ApiResponse<Long> create(@RequestHeader("X-User-ID") Long userId,
            @RequestBody OrderCreateRequest request) {
        Long orderId = orderService.createOrder(userId, request.getRemark(), request.getItems());
        return ApiResponse.success(orderId);
    }

    @GetMapping("/{id}")
    public ApiResponse<OrderWithItems> detail(@RequestHeader("X-User-ID") Long userId,
            @PathVariable("id") Long id) {
        var opt = orderService.getOrderDetail(id, userId).map(ApiResponse::success);
        return opt.orElseGet(() -> new ApiResponse<>(ResponseStatus.ERROR, "Not found"));
    }

    @GetMapping("/list")
    public ApiResponse<?> list(@RequestHeader("X-User-ID") Long userId,
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize) {
        var orders = orderService.listOrdersByUser(userId, pageNum, pageSize);
        return ApiResponse.success(orders);
    }

    @PostMapping("/cancel/{id}")
    public ApiResponse<?> cancel(@RequestHeader("X-User-ID") Long userId,
            @PathVariable("id") Long id) {
        boolean ok = orderService.cancelOrder(id, userId);
        if (ok) {
            return ApiResponse.success();
        } else {
            return new ApiResponse<>(ResponseStatus.ERROR, "Cancel failed");
        }
    }

    @GetMapping("/micro-service-test")
    public String microServiceTest() {
        return "No implementation";
    }
}
