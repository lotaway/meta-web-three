package com.metawebthree.order.interfaces.web;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.order.domain.model.OrderReturnReasonDO;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderReturnReasonMapper;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@Validated
@RestController
@RequestMapping("/returnReason")
@RequiredArgsConstructor
@Tag(name = "Order Return Reason Management")
public class OrderReturnReasonController {

    private final OrderReturnReasonMapper reasonMapper;

    @Operation(summary = "分页查询退货原因")
    @GetMapping("/list")
    public ApiResponse<Page<OrderReturnReasonDO>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize) {
        return ApiResponse.success(reasonMapper.selectPage(new Page<>(pageNum, pageSize),
                new LambdaQueryWrapper<OrderReturnReasonDO>().orderByDesc(OrderReturnReasonDO::getId)));
    }

    @Operation(summary = "添加退货原因")
    @PostMapping("/create")
    public ApiResponse<Void> create(@RequestBody OrderReturnReasonDO reason) {
        reason.setCreateTime(LocalDateTime.now());
        reasonMapper.insert(reason);
        return ApiResponse.success();
    }

    @Operation(summary = "修改退货原因")
    @PostMapping("/update/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody OrderReturnReasonDO reason) {
        reason.setId(id);
        reasonMapper.updateById(reason);
        return ApiResponse.success();
    }

    @Operation(summary = "批量删除退货原因")
    @PostMapping("/delete")
    public ApiResponse<Void> delete(@RequestParam String ids) {
        List<Long> idList = Arrays.stream(ids.split(","))
                .map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        reasonMapper.deleteByIds(idList);
        return ApiResponse.success();
    }

    @Operation(summary = "修改退货原因启用状态")
    @PostMapping("/update/status")
    public ApiResponse<Void> updateStatus(@RequestParam String ids, @RequestParam Integer status) {
        List<Long> idList = Arrays.stream(ids.split(","))
                .map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        reasonMapper.update(null, new UpdateWrapper<OrderReturnReasonDO>()
                .in("id", idList).set("status", status));
        return ApiResponse.success();
    }

    @Operation(summary = "获取单个退货原因详情")
    @GetMapping("/{id}")
    public ApiResponse<OrderReturnReasonDO> getById(@PathVariable Long id) {
        return ApiResponse.success(reasonMapper.selectById(id));
    }
}
