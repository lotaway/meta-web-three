package com.metawebthree.promotion.interfaces.web;

import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.promotion.infrastructure.persistence.mapper.FlashPromotionMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.FlashPromotionDO;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

@Validated
@RestController
@RequestMapping("/flash")
@RequiredArgsConstructor
@Tag(name = "Flash Promotion Admin")
public class FlashPromotionController {

    private final FlashPromotionMapper flashPromotionMapper;

    @Operation(summary = "分页查询秒杀活动")
    @GetMapping("/list")
    public ApiResponse<Page<FlashPromotionDO>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String keyword) {
        return ApiResponse.success(flashPromotionMapper.selectPage(new Page<>(pageNum, pageSize), null));
    }

    @Operation(summary = "修改活动上下线状态")
    @PostMapping("/update/status/{id}")
    public ApiResponse<Void> updateStatus(@PathVariable Long id, @RequestParam Integer status) {
        flashPromotionMapper.update(null, new UpdateWrapper<FlashPromotionDO>().eq("id", id).set("status", status));
        return ApiResponse.success();
    }

    @Operation(summary = "删除活动")
    @PostMapping("/delete/{id}")
    public ApiResponse<Void> delete(@PathVariable Long id) {
        flashPromotionMapper.deleteById(id);
        return ApiResponse.success();
    }

    @Operation(summary = "添加活动")
    @PostMapping("/create")
    public ApiResponse<Void> create(@RequestBody FlashPromotionDO promotion) {
        flashPromotionMapper.insert(promotion);
        return ApiResponse.success();
    }

    @Operation(summary = "修改活动")
    @PostMapping("/update/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody FlashPromotionDO promotion) {
        promotion.setId(id);
        flashPromotionMapper.updateById(promotion);
        return ApiResponse.success();
    }
}
