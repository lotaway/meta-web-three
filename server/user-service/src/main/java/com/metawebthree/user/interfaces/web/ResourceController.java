package com.metawebthree.user.interfaces.web;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.user.application.ResourceService;
import com.metawebthree.user.domain.model.ResourceDO;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;

@RestController
@RequestMapping("/resource")
@RequiredArgsConstructor
@Tag(name = "Resource Controller", description = "后台资源管理")
public class ResourceController {

    private final ResourceService resourceService;

    @Operation(summary = "查询所有资源")
    @GetMapping("/listAll")
    @RequirePermission("ums:resource:read")
    public ApiResponse<List<ResourceDO>> listAll() {
        return ApiResponse.success(resourceService.listAll());
    }

    @Operation(summary = "分页查询资源")
    @GetMapping("/list")
    @RequirePermission("ums:resource:read")
    public ApiResponse<Page<ResourceDO>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String nameKeyword,
            @RequestParam(required = false) String urlKeyword,
            @RequestParam(required = false) Long categoryId) {
        return ApiResponse.success(resourceService.listResources(pageNum, pageSize, nameKeyword, urlKeyword, categoryId));
    }

    @Operation(summary = "添加资源")
    @PostMapping("/create")
    @RequirePermission("ums:resource:create")
    public ApiResponse<Void> create(@RequestBody ResourceDO resource) {
        resource.setCreateTime(LocalDateTime.now());
        resourceService.save(resource);
        return ApiResponse.success();
    }

    @Operation(summary = "修改资源")
    @PostMapping("/update/{id}")
    @RequirePermission("ums:resource:update")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody ResourceDO resource) {
        resource.setId(id);
        resourceService.updateById(resource);
        return ApiResponse.success();
    }

    @Operation(summary = "删除资源")
    @PostMapping("/delete/{id}")
    @RequirePermission("ums:resource:delete")
    public ApiResponse<Void> delete(@PathVariable Long id) {
        resourceService.removeById(id);
        return ApiResponse.success();
    }
}
