package com.metawebthree.order.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.order.application.OrderReturnApplicationService;
import com.metawebthree.order.domain.model.OrderReturnApply;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@Validated
@RestController
@RequestMapping("/v1/order-returns")
@RequiredArgsConstructor
@Tag(name = "Order Return Management")
public class OrderReturnController {

    private final OrderReturnApplicationService returnService;

    @Operation(summary = "Submit return application")
    @PostMapping
    public ApiResponse<Void> apply(@RequestBody OrderReturnApply apply) {
        returnService.applyReturn(apply);
        return ApiResponse.success();
    }

    @Operation(summary = "Get return application by id")
    @GetMapping("/{id}")
    public ApiResponse<OrderReturnApply> details(@PathVariable Long id) {
        return ApiResponse.success(returnService.getReturn(id));
    }

    @Operation(summary = "Update return application status")
    @PutMapping("/{id}")
    public ApiResponse<Void> handle(@PathVariable Long id, @RequestBody OrderReturnApply apply) {
        returnService.handleReturn(apply);
        return ApiResponse.success();
    }

    @Operation(summary = "List return applications by order Sn")
    @GetMapping("/order/{orderSn}")
    public ApiResponse<List<OrderReturnApply>> listByOrder(@PathVariable String orderSn) {
        return ApiResponse.success(returnService.listByOrder(orderSn));
    }

    @Operation(summary = "Remove application history")
    @DeleteMapping("/{id}")
    public ApiResponse<Void> remove(@PathVariable Long id) {
        returnService.removeHistory(id);
        return ApiResponse.success();
    }
}
