package com.metawebthree.promotion.interfaces.web;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.promotion.infrastructure.persistence.mapper.AdvertiseMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.AdvertiseRecord;
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
@RequestMapping("/home/advertise")
@RequiredArgsConstructor
@Tag(name = "Home Advertise Admin")
public class HomeAdvertiseController {

    private final AdvertiseMapper advertiseMapper;

    @Operation(summary = "分页查询广告")
    @GetMapping("/list")
    public ApiResponse<Page<AdvertiseRecord>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String name,
            @RequestParam(required = false) Integer type,
            @RequestParam(required = false) Integer status) {
        LambdaQueryWrapper<AdvertiseRecord> wrapper = new LambdaQueryWrapper<AdvertiseRecord>().orderByDesc(AdvertiseRecord::getId);
        if (name != null && !name.isEmpty()) wrapper.like(AdvertiseRecord::getName, name);
        if (type != null) wrapper.eq(AdvertiseRecord::getType, type);
        if (status != null) wrapper.eq(AdvertiseRecord::getStatus, status);
        return ApiResponse.success(advertiseMapper.selectPage(new Page<>(pageNum, pageSize), wrapper));
    }

    @Operation(summary = "修改上下线状态")
    @PostMapping("/update/status/{id}")
    public ApiResponse<Void> updateStatus(@PathVariable Long id, @RequestParam Integer status) {
        advertiseMapper.update(null, new UpdateWrapper<AdvertiseRecord>().eq("id", id).set("status", status));
        return ApiResponse.success();
    }

    @Operation(summary = "批量删除广告")
    @PostMapping("/delete")
    public ApiResponse<Void> delete(@RequestParam String ids) {
        List<Long> idList = Arrays.stream(ids.split(","))
                .map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        advertiseMapper.deleteByIds(idList);
        return ApiResponse.success();
    }

    @Operation(summary = "添加广告")
    @PostMapping("/create")
    public ApiResponse<Void> create(@RequestBody AdvertiseRecord advertise) {
        advertiseMapper.insert(advertise);
        return ApiResponse.success();
    }

    @Operation(summary = "获取广告详情")
    @GetMapping("/{id}")
    public ApiResponse<AdvertiseRecord> getById(@PathVariable Long id) {
        return ApiResponse.success(advertiseMapper.selectById(id));
    }

    @Operation(summary = "修改广告")
    @PostMapping("/update/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody AdvertiseRecord advertise) {
        advertise.setId(id);
        advertiseMapper.updateById(advertise);
        return ApiResponse.success();
    }
}
