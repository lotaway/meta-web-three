package com.metawebthree.user.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.user.application.ResourceCategoryService;
import com.metawebthree.user.domain.model.ResourceCategoryDO;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;

@RestController
@RequestMapping("/resourceCategory")
@RequiredArgsConstructor
@Tag(name = "Resource Category Controller", description = "后台资源分类管理")
public class ResourceCategoryController {

    private final ResourceCategoryService resourceCategoryService;

    @Operation(summary = "查询所有资源分类")
    @GetMapping("/listAll")
    public ApiResponse<List<ResourceCategoryDO>> listAll() {
        return ApiResponse.success(resourceCategoryService.listAll());
    }

    @Operation(summary = "添加资源分类")
    @PostMapping("/create")
    public ApiResponse<Void> create(@RequestBody ResourceCategoryDO category) {
        category.setCreateTime(LocalDateTime.now());
        resourceCategoryService.save(category);
        return ApiResponse.success();
    }

    @Operation(summary = "修改资源分类")
    @PostMapping("/update/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody ResourceCategoryDO category) {
        category.setId(id);
        resourceCategoryService.updateById(category);
        return ApiResponse.success();
    }

    @Operation(summary = "删除资源分类")
    @PostMapping("/delete/{id}")
    public ApiResponse<Void> delete(@PathVariable Long id) {
        resourceCategoryService.removeById(id);
        return ApiResponse.success();
    }
}
