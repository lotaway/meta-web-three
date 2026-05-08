package com.metawebthree.promotion.interfaces.web;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.promotion.infrastructure.persistence.mapper.FlashPromotionProductRelationMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.FlashPromotionProductRelationDO;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@Validated
@RestController
@RequestMapping("/flashProductRelation")
@RequiredArgsConstructor
@Tag(name = "Flash Promotion Product Relation Admin")
public class FlashPromotionProductRelationController {

    private final FlashPromotionProductRelationMapper relationMapper;

    @Operation(summary = "分页查询不同场次关联及商品信息")
    @GetMapping("/list")
    public ApiResponse<Page<FlashPromotionProductRelationDO>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) Long flashPromotionId,
            @RequestParam(required = false) Long flashPromotionSessionId) {
        LambdaQueryWrapper<FlashPromotionProductRelationDO> wrapper = new LambdaQueryWrapper<FlashPromotionProductRelationDO>().orderByDesc(FlashPromotionProductRelationDO::getId);
        if (flashPromotionId != null) wrapper.eq(FlashPromotionProductRelationDO::getFlashPromotionId, flashPromotionId);
        if (flashPromotionSessionId != null) wrapper.eq(FlashPromotionProductRelationDO::getFlashPromotionSessionId, flashPromotionSessionId);
        return ApiResponse.success(relationMapper.selectPage(new Page<>(pageNum, pageSize), wrapper));
    }

    @Operation(summary = "批量选择商品添加关联")
    @PostMapping("/create")
    public ApiResponse<Void> create(@RequestBody List<FlashPromotionProductRelationDO> relations) {
        for (FlashPromotionProductRelationDO r : relations) {
            relationMapper.insert(r);
        }
        return ApiResponse.success();
    }

    @Operation(summary = "删除关联")
    @PostMapping("/delete/{id}")
    public ApiResponse<Void> delete(@PathVariable Long id) {
        relationMapper.deleteById(id);
        return ApiResponse.success();
    }

    @Operation(summary = "修改关联")
    @PostMapping("/update/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody FlashPromotionProductRelationDO relation) {
        relation.setId(id);
        relationMapper.updateById(relation);
        return ApiResponse.success();
    }
}
