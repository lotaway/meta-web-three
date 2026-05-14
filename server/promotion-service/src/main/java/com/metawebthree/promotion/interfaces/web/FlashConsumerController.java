package com.metawebthree.promotion.interfaces.web;

import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.promotion.application.service.FlashConsumerService;
import com.metawebthree.promotion.interfaces.web.dto.FlashOrderRequest;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/flash")
@RequiredArgsConstructor
@Tag(name = "Flash Consumer", description = "消费者端闪购")
public class FlashConsumerController {

    private final FlashConsumerService flashConsumerService;

    @Operation(summary = "获取当前闪购活动及场次商品")
    @GetMapping("/current")
    public ApiResponse<List<Map<String, Object>>> getCurrentPromotions() {
        return ApiResponse.success(flashConsumerService.getCurrentPromotions());
    }

    @Operation(summary = "获取场次内闪购商品")
    @GetMapping("/current/products")
    public ApiResponse<List<Map<String, Object>>> getCurrentSessionProducts(@RequestParam Long sessionId) {
        return ApiResponse.success(flashConsumerService.getCurrentSessionProducts(sessionId));
    }

    @Operation(summary = "查询商品闪购信息")
    @GetMapping("/product/{productId}")
    public ApiResponse<Map<String, Object>> getProductFlashInfo(@PathVariable Long productId) {
        return ApiResponse.success(flashConsumerService.getProductFlashInfo(productId));
    }

    @Operation(summary = "闪购下单")
    @PostMapping("/order")
    public ApiResponse<Map<String, Object>> createFlashOrder(
            @RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestBody FlashOrderRequest request) {
        try {
            Long orderId = flashConsumerService.createFlashOrder(userId, request);
            return ApiResponse.success(Map.of("orderId", orderId));
        } catch (IllegalArgumentException e) {
            return ApiResponse.error(ResponseStatus.PARAM_ERROR, e.getMessage());
        } catch (IllegalStateException e) {
            return ApiResponse.error(ResponseStatus.SYSTEM_ERROR, e.getMessage());
        }
    }
}
