package com.metawebthree.promotion.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.promotion.application.AdvertiseService;
import com.metawebthree.promotion.domain.model.Advertise;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Validated
@RestController
@RequestMapping("/v1/promotion/advertises")
@RequiredArgsConstructor
@Tag(name = "Advertise Management", description = "广告管理接口")
public class AdvertiseController {

    private final AdvertiseService advertiseService;

    @Operation(summary = "添加广告")
    @PostMapping
    public ApiResponse<Void> create(@RequestBody Advertise advertise) {
        advertiseService.create(advertise);
        return ApiResponse.success();
    }

    @Operation(summary = "更新广告")
    @PutMapping("/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody Advertise advertise) {
        advertiseService.update(advertise);
        return ApiResponse.success();
    }

    @Operation(summary = "删除广告")
    @DeleteMapping("/{id}")
    public ApiResponse<Void> delete(@PathVariable Long id) {
        advertiseService.delete(id);
        return ApiResponse.success();
    }

    @Operation(summary = "获取广告详情")
    @GetMapping("/{id}")
    public ApiResponse<Advertise> getById(@PathVariable Long id) {
        return ApiResponse.success(advertiseService.getById(id));
    }

    @Operation(summary = "分页查询广告")
    @GetMapping
    public ApiResponse<List<Advertise>> list(
            @RequestParam(required = false) String name,
            @RequestParam(required = false) Integer type,
            @RequestParam(required = false) String endTime,
            @RequestParam(required = false) Integer status) {
        return ApiResponse.success(advertiseService.list(name, type, endTime, status));
    }

    @Operation(summary = "获取可用广告列表")
    @GetMapping("/available")
    public ApiResponse<List<Advertise>> listAvailable(@RequestParam(defaultValue = "1") Integer type) {
        return ApiResponse.success(advertiseService.listAvailable(type));
    }
}
