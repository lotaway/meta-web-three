package com.metawebthree.promotion.interfaces.web;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.promotion.infrastructure.persistence.mapper.CouponDOMapper;
import com.metawebthree.promotion.infrastructure.persistence.mapper.CouponHistoryMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.CouponDO;
import com.metawebthree.promotion.infrastructure.persistence.model.CouponHistoryDO;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

@Validated
@RestController
@RequiredArgsConstructor
@Tag(name = "Coupon Admin")
public class CouponAdminController {

    private final CouponDOMapper couponMapper;
    private final CouponHistoryMapper couponHistoryMapper;

    @Operation(summary = "分页获取优惠券列表")
    @GetMapping("/coupon/list")
    public ApiResponse<Page<CouponDO>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String name,
            @RequestParam(required = false) Integer type) {
        LambdaQueryWrapper<CouponDO> wrapper = new LambdaQueryWrapper<CouponDO>().orderByDesc(CouponDO::getId);
        if (name != null && !name.isEmpty()) wrapper.like(CouponDO::getName, name);
        if (type != null) wrapper.eq(CouponDO::getType, type);
        return ApiResponse.success(couponMapper.selectPage(new Page<>(pageNum, pageSize), wrapper));
    }

    @Operation(summary = "添加优惠券")
    @PostMapping("/coupon/create")
    public ApiResponse<Void> create(@RequestBody CouponDO coupon) {
        couponMapper.insert(coupon);
        return ApiResponse.success();
    }

    @Operation(summary = "获取单个优惠券详细信息")
    @GetMapping("/coupon/{id}")
    public ApiResponse<CouponDO> getById(@PathVariable Long id) {
        return ApiResponse.success(couponMapper.selectById(id));
    }

    @Operation(summary = "修改优惠券")
    @PostMapping("/coupon/update/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody CouponDO coupon) {
        coupon.setId(id);
        couponMapper.updateById(coupon);
        return ApiResponse.success();
    }

    @Operation(summary = "删除优惠券")
    @PostMapping("/coupon/delete/{id}")
    public ApiResponse<Void> delete(@PathVariable Long id) {
        couponMapper.deleteById(id);
        return ApiResponse.success();
    }

    @Operation(summary = "分页获取优惠券历史记录")
    @GetMapping("/couponHistory/list")
    public ApiResponse<Page<CouponHistoryDO>> historyList(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) Long couponId,
            @RequestParam(required = false) Integer useStatus,
            @RequestParam(required = false) String orderSn) {
        LambdaQueryWrapper<CouponHistoryDO> wrapper = new LambdaQueryWrapper<CouponHistoryDO>().orderByDesc(CouponHistoryDO::getId);
        if (couponId != null) wrapper.eq(CouponHistoryDO::getCouponId, couponId);
        if (useStatus != null) wrapper.eq(CouponHistoryDO::getUseStatus, useStatus);
        if (orderSn != null && !orderSn.isEmpty()) wrapper.eq(CouponHistoryDO::getOrderSn, orderSn);
        return ApiResponse.success(couponHistoryMapper.selectPage(new Page<>(pageNum, pageSize), wrapper));
    }
}
