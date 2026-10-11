package com.metawebthree.promotion.interfaces.web;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.promotion.application.service.HomeRecommendSubjectService;
import com.metawebthree.promotion.infrastructure.persistence.model.HomeRecommendSubjectDO;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@Validated
@RestController
@RequestMapping("/home/subject")
@RequiredArgsConstructor
@Tag(name = "Home Subject Admin")
public class HomeSubjectAdminController {

    private final HomeRecommendSubjectService homeSubjectService;

    @Operation(summary = "分页查询首页专题推荐")
    @GetMapping("/list")
    public ApiResponse<Page<HomeRecommendSubjectDO>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String subjectName,
            @RequestParam(required = false) Integer recommendStatus) {
        return ApiResponse.success(homeSubjectService.list(pageNum, pageSize, subjectName, recommendStatus));
    }

    @Operation(summary = "批量修改推荐状态")
    @PostMapping("/update/recommendStatus")
    public ApiResponse<Void> updateRecommendStatus(@RequestParam String ids, @RequestParam Integer recommendStatus) {
        homeSubjectService.updateRecommendStatus(ids, recommendStatus);
        return ApiResponse.success();
    }

    @Operation(summary = "批量删除推荐")
    @PostMapping("/delete")
    public ApiResponse<Void> delete(@RequestParam String ids) {
        homeSubjectService.delete(ids);
        return ApiResponse.success();
    }

    @Operation(summary = "批量添加推荐")
    @PostMapping("/create")
    public ApiResponse<Void> create(@RequestBody List<HomeRecommendSubjectDO> subjects) {
        homeSubjectService.create(subjects);
        return ApiResponse.success();
    }

    @Operation(summary = "修改排序")
    @PostMapping("/update/sort/{id}")
    public ApiResponse<Void> updateSort(@PathVariable Long id, @RequestParam Integer sort) {
        homeSubjectService.updateSort(id, sort);
        return ApiResponse.success();
    }
}
