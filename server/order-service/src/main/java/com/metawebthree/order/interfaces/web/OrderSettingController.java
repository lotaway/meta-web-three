package com.metawebthree.order.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.order.application.OrderSettingApplicationService;
import com.metawebthree.order.domain.model.OrderSetting;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

@Validated
@RestController
@RequestMapping("/v1/order-settings")
@RequiredArgsConstructor
@Tag(name = "Order Setting Management")
public class OrderSettingController {

    private final OrderSettingApplicationService settingService;

    @Operation(summary = "Set global order timeouts")
    @PostMapping
    public ApiResponse<Void> define(@RequestBody OrderSetting setting) {
        settingService.defineSetting(setting);
        return ApiResponse.success();
    }

    @Operation(summary = "Get global order setting")
    @GetMapping
    public ApiResponse<OrderSetting> get() {
        return ApiResponse.success(settingService.getSetting());
    }

    @Operation(summary = "Update global order setting")
    @PutMapping("/{id}")
    public ApiResponse<Void> modify(@PathVariable Long id, @RequestBody OrderSetting setting) {
        settingService.updateSetting(setting);
        return ApiResponse.success();
    }
}
