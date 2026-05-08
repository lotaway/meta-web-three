package com.metawebthree.promotion.interfaces.web;

import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.promotion.infrastructure.persistence.mapper.HomeBrandMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.HomeBrandDO;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@Validated
@RestController
@RequestMapping("/home/brand")
@RequiredArgsConstructor
@Tag(name = "Home Brand Admin")
public class HomeBrandAdminController {

    private final HomeBrandMapper homeBrandMapper;

    @Operation(summary = "分页查询推荐品牌")
    @GetMapping("/list")
    public ApiResponse<Page<HomeBrandDO>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String brandName,
            @RequestParam(required = false) Integer recommendStatus) {
        return ApiResponse.success(homeBrandMapper.selectPage(new Page<>(pageNum, pageSize), null));
    }

    @Operation(summary = "批量修改推荐品牌状态")
    @PostMapping("/update/recommendStatus")
    public ApiResponse<Void> updateRecommendStatus(@RequestParam String ids, @RequestParam Integer recommendStatus) {
        List<Long> idList = Arrays.stream(ids.split(",")).map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        homeBrandMapper.update(null, new UpdateWrapper<HomeBrandDO>().in("id", idList).set("recommend_status", recommendStatus));
        return ApiResponse.success();
    }

    @Operation(summary = "批量删除推荐品牌")
    @PostMapping("/delete")
    public ApiResponse<Void> delete(@RequestParam String ids) {
        List<Long> idList = Arrays.stream(ids.split(",")).map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        homeBrandMapper.deleteByIds(idList);
        return ApiResponse.success();
    }

    @Operation(summary = "添加首页推荐品牌")
    @PostMapping("/create")
    public ApiResponse<Void> create(@RequestBody List<HomeBrandDO> brands) {
        for (HomeBrandDO b : brands) homeBrandMapper.insert(b);
        return ApiResponse.success();
    }

    @Operation(summary = "修改推荐品牌排序")
    @PostMapping("/update/sort/{id}")
    public ApiResponse<Void> updateSort(@PathVariable Long id, @RequestParam Integer sort) {
        homeBrandMapper.update(null, new UpdateWrapper<HomeBrandDO>().eq("id", id).set("sort", sort));
        return ApiResponse.success();
    }
}
