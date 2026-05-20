package com.metawebthree.promotion.interfaces.web;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.promotion.application.service.HomeNewProductService;
import com.metawebthree.promotion.infrastructure.persistence.model.HomeNewProductDO;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@Validated
@RestController
@RequestMapping("/home/newProduct")
@RequiredArgsConstructor
@Tag(name = "Home New Product Admin")
public class HomeNewProductAdminController {

    private final HomeNewProductService newProductService;

    @Operation(summary = "分页查询首页新品")
    @GetMapping("/list")
    public ApiResponse<Page<HomeNewProductDO>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String productName,
            @RequestParam(required = false) Integer recommendStatus) {
        return ApiResponse.success(newProductService.list(pageNum, pageSize, productName, recommendStatus));
    }

    @Operation(summary = "批量修改首页新品状态")
    @PostMapping("/update/recommendStatus")
    public ApiResponse<Void> updateRecommendStatus(@RequestParam String ids, @RequestParam Integer recommendStatus) {
        newProductService.updateRecommendStatus(ids, recommendStatus);
        return ApiResponse.success();
    }

    @Operation(summary = "批量删除首页新品")
    @PostMapping("/delete")
    public ApiResponse<Void> delete(@RequestParam String ids) {
        newProductService.delete(ids);
        return ApiResponse.success();
    }

    @Operation(summary = "批量添加首页新品")
    @PostMapping("/create")
    public ApiResponse<Void> create(@RequestBody List<HomeNewProductDO> products) {
        newProductService.create(products);
        return ApiResponse.success();
    }

    @Operation(summary = "修改首页新品排序")
    @PostMapping("/update/sort/{id}")
    public ApiResponse<Void> updateSort(@PathVariable Long id, @RequestParam Integer sort) {
        newProductService.updateSort(id, sort);
        return ApiResponse.success();
    }
}
