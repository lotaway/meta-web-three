package com.metawebthree.promotion.interfaces.web;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.promotion.application.service.HomeBrandService;
import com.metawebthree.promotion.infrastructure.persistence.model.HomeBrandDO;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@Validated
@RestController
@RequestMapping("/home/brand")
@RequiredArgsConstructor
@Tag(name = "Home Brand Admin")
public class HomeBrandAdminController {

    private final HomeBrandService homeBrandService;

    @Operation(summary = "分页查询推荐品牌")
    @GetMapping("/list")
    public ApiResponse<Page<HomeBrandDO>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String brandName,
            @RequestParam(required = false) Integer recommendStatus) {
        return ApiResponse.success(homeBrandService.list(pageNum, pageSize, brandName, recommendStatus));
    }

    @Operation(summary = "批量修改推荐品牌状态")
    @PostMapping("/update/recommendStatus")
    public ApiResponse<Void> updateRecommendStatus(@RequestParam String ids, @RequestParam Integer recommendStatus) {
        homeBrandService.updateRecommendStatus(ids, recommendStatus);
        return ApiResponse.success();
    }

    @Operation(summary = "批量删除推荐品牌")
    @PostMapping("/delete")
    public ApiResponse<Void> delete(@RequestParam String ids) {
        homeBrandService.delete(ids);
        return ApiResponse.success();
    }

    @Operation(summary = "添加首页推荐品牌")
    @PostMapping("/create")
    public ApiResponse<Void> create(@RequestBody List<HomeBrandDO> brands) {
        homeBrandService.create(brands);
        return ApiResponse.success();
    }

    @Operation(summary = "修改推荐品牌排序")
    @PostMapping("/update/sort/{id}")
    public ApiResponse<Void> updateSort(@PathVariable Long id, @RequestParam Integer sort) {
        homeBrandService.updateSort(id, sort);
        return ApiResponse.success();
    }
}
