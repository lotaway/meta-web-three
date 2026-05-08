package com.metawebthree.promotion.interfaces.web;

import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.promotion.infrastructure.persistence.mapper.HomeRecommendProductMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.HomeRecommendProductDO;
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
@RequestMapping("/home/recommendProduct")
@RequiredArgsConstructor
@Tag(name = "Home Recommend Product Admin")
public class HomeRecommendProductAdminController {

    private final HomeRecommendProductMapper recommendProductMapper;

    @Operation(summary = "分页查询人气推荐")
    @GetMapping("/list")
    public ApiResponse<Page<HomeRecommendProductDO>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String productName,
            @RequestParam(required = false) Integer recommendStatus) {
        return ApiResponse.success(recommendProductMapper.selectPage(new Page<>(pageNum, pageSize), null));
    }

    @Operation(summary = "批量修改推荐状态")
    @PostMapping("/update/recommendStatus")
    public ApiResponse<Void> updateRecommendStatus(@RequestParam String ids, @RequestParam Integer recommendStatus) {
        List<Long> idList = Arrays.stream(ids.split(",")).map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        recommendProductMapper.update(null, new UpdateWrapper<HomeRecommendProductDO>().in("id", idList).set("recommend_status", recommendStatus));
        return ApiResponse.success();
    }

    @Operation(summary = "批量删除推荐")
    @PostMapping("/delete")
    public ApiResponse<Void> delete(@RequestParam String ids) {
        List<Long> idList = Arrays.stream(ids.split(",")).map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        recommendProductMapper.deleteByIds(idList);
        return ApiResponse.success();
    }

    @Operation(summary = "批量添加推荐")
    @PostMapping("/create")
    public ApiResponse<Void> create(@RequestBody List<HomeRecommendProductDO> products) {
        for (HomeRecommendProductDO p : products) recommendProductMapper.insert(p);
        return ApiResponse.success();
    }

    @Operation(summary = "修改排序")
    @PostMapping("/update/sort/{id}")
    public ApiResponse<Void> updateSort(@PathVariable Long id, @RequestParam Integer sort) {
        recommendProductMapper.update(null, new UpdateWrapper<HomeRecommendProductDO>().eq("id", id).set("sort", sort));
        return ApiResponse.success();
    }
}
