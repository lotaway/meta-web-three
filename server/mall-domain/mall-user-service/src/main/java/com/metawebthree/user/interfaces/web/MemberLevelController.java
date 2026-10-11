package com.metawebthree.user.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.user.application.MemberLevelService;
import com.metawebthree.user.domain.model.MemberLevelDO;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/memberLevel")
@RequiredArgsConstructor
@Tag(name = "Member Level Controller", description = "会员等级管理")
public class MemberLevelController {

    private final MemberLevelService memberLevelService;

    @Operation(summary = "查询所有会员等级")
    @GetMapping("/list")
    public ApiResponse<List<MemberLevelDO>> list(@RequestParam(required = false) Integer defaultStatus) {
        return ApiResponse.success(memberLevelService.listByDefaultStatus(defaultStatus));
    }

    @Operation(summary = "获取会员等级详情")
    @GetMapping("/{id}")
    public ApiResponse<MemberLevelDO> getById(@PathVariable Long id) {
        MemberLevelDO memberLevel = memberLevelService.getById(id);
        return ApiResponse.success(memberLevel);
    }

    @Operation(summary = "创建会员等级")
    @PostMapping
    public ApiResponse<Long> create(@RequestBody MemberLevelDO memberLevel) {
        memberLevelService.save(memberLevel);
        return ApiResponse.success(memberLevel.getId());
    }

    @Operation(summary = "更新会员等级")
    @PutMapping("/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody MemberLevelDO memberLevel) {
        memberLevel.setId(id);
        memberLevelService.updateById(memberLevel);
        return ApiResponse.success(null);
    }

    @Operation(summary = "删除会员等级")
    @DeleteMapping("/{id}")
    public ApiResponse<Void> delete(@PathVariable Long id) {
        memberLevelService.removeById(id);
        return ApiResponse.success(null);
    }
}
