package com.metawebthree.promotion.interfaces.web;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.promotion.application.service.FlashPromotionProductRelationService;
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

    private final FlashPromotionProductRelationService relationService;

    @Operation(summary = "分页查询不同场次关联及商品信息")
    @GetMapping("/list")
    public ApiResponse<Page<FlashPromotionProductRelationDO>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) Long flashPromotionId,
            @RequestParam(required = false) Long flashPromotionSessionId) {
        return ApiResponse.success(relationService.list(pageNum, pageSize, flashPromotionId, flashPromotionSessionId));
    }

    @Operation(summary = "批量选择商品添加关联")
    @PostMapping("/create")
    public ApiResponse<Void> create(@RequestBody List<FlashPromotionProductRelationDO> relations) {
        relationService.create(relations);
        return ApiResponse.success();
    }

    @Operation(summary = "删除关联")
    @PostMapping("/delete/{id}")
    public ApiResponse<Void> delete(@PathVariable Long id) {
        relationService.delete(id);
        return ApiResponse.success();
    }

    @Operation(summary = "修改关联")
    @PostMapping("/update/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody FlashPromotionProductRelationDO relation) {
        relationService.update(id, relation);
        return ApiResponse.success();
    }
}
