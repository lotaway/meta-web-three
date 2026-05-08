package com.metawebthree.promotion.interfaces.web;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.promotion.application.service.CmsSubjectService;
import com.metawebthree.promotion.infrastructure.persistence.model.CmsSubjectDO;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@Validated
@RestController
@RequestMapping("/subject")
@RequiredArgsConstructor
@Tag(name = "CMS Subject Admin")
public class CmsSubjectController {
    private final CmsSubjectService subjectService;

    @Operation(summary = "获取全部商品专题")
    @GetMapping("/listAll")
    public ApiResponse<List<CmsSubjectDO>> listAll() {
        return ApiResponse.success(subjectService.listAll());
    }

    @Operation(summary = "根据专题名称分页获取商品专题")
    @GetMapping("/list")
    public ApiResponse<Page<CmsSubjectDO>> list(
            @RequestParam(required = false) String keyword,
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "5") Integer pageSize) {
        return ApiResponse.success(subjectService.list(keyword, pageNum, pageSize));
    }

    @Operation(summary = "获取专题详情")
    @GetMapping("/{id}")
    public ApiResponse<CmsSubjectDO> getById(@PathVariable Long id) {
        return ApiResponse.success(subjectService.getById(id));
    }

    @Operation(summary = "添加专题")
    @PostMapping("/create")
    public ApiResponse<Void> create(@RequestBody CmsSubjectDO subject) {
        subjectService.create(subject);
        return ApiResponse.success();
    }

    @Operation(summary = "修改专题")
    @PostMapping("/update/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody CmsSubjectDO subject) {
        subjectService.update(id, subject);
        return ApiResponse.success();
    }

    @Operation(summary = "删除专题")
    @PostMapping("/delete/{id}")
    public ApiResponse<Void> delete(@PathVariable Long id) {
        subjectService.delete(id);
        return ApiResponse.success();
    }
}
