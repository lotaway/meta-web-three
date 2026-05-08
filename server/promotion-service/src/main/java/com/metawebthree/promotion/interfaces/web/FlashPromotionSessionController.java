package com.metawebthree.promotion.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.promotion.application.service.FlashPromotionSessionService;
import com.metawebthree.promotion.infrastructure.persistence.model.FlashPromotionSessionDO;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@Validated
@RestController
@RequestMapping("/flashSession")
@RequiredArgsConstructor
@Tag(name = "Flash Promotion Session Admin")
public class FlashPromotionSessionController {

    private final FlashPromotionSessionService sessionService;

    @Operation(summary = "获取全部场次")
    @GetMapping("/list")
    public ApiResponse<List<FlashPromotionSessionDO>> list() {
        return ApiResponse.success(sessionService.list());
    }

    @Operation(summary = "获取全部可选场次及其数量")
    @GetMapping("/selectList")
    public ApiResponse<List<FlashPromotionSessionDO>> selectList(@RequestParam(required = false) Long flashPromotionId) {
        return ApiResponse.success(sessionService.selectList(flashPromotionId));
    }

    @Operation(summary = "修改场次启用状态")
    @PostMapping("/update/status/{id}")
    public ApiResponse<Void> updateStatus(@PathVariable Long id, @RequestParam Integer status) {
        sessionService.updateStatus(id, status);
        return ApiResponse.success();
    }

    @Operation(summary = "删除场次")
    @PostMapping("/delete/{id}")
    public ApiResponse<Void> delete(@PathVariable Long id) {
        sessionService.delete(id);
        return ApiResponse.success();
    }

    @Operation(summary = "添加场次")
    @PostMapping("/create")
    public ApiResponse<Void> create(@RequestBody FlashPromotionSessionDO session) {
        sessionService.create(session);
        return ApiResponse.success();
    }

    @Operation(summary = "修改场次")
    @PostMapping("/update/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody FlashPromotionSessionDO session) {
        sessionService.update(id, session);
        return ApiResponse.success();
    }
}
