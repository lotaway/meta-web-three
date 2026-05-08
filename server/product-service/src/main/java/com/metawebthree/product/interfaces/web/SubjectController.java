package com.metawebthree.product.interfaces.web;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.product.domain.model.SubjectDO;
import com.metawebthree.product.infrastructure.persistence.mapper.SubjectMapper;
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
@Tag(name = "Subject Management")
public class SubjectController {

    private final SubjectMapper subjectMapper;

    @Operation(summary = "获取全部专题列表")
    @GetMapping("/listAll")
    public ApiResponse<List<SubjectDO>> listAll() {
        return ApiResponse.success(subjectMapper.selectList(null));
    }

    @Operation(summary = "分页查询专题")
    @GetMapping("/list")
    public ApiResponse<Page<SubjectDO>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String keyword) {
        LambdaQueryWrapper<SubjectDO> wrapper = new LambdaQueryWrapper<SubjectDO>().orderByDesc(SubjectDO::getId);
        if (keyword != null && !keyword.isEmpty()) {
            wrapper.like(SubjectDO::getTitle, keyword);
        }
        return ApiResponse.success(subjectMapper.selectPage(new Page<>(pageNum, pageSize), wrapper));
    }
}
