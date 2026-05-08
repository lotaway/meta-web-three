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
}
