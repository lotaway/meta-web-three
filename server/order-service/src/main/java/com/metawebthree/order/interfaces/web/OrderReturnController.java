package com.metawebthree.order.interfaces.web;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.order.application.OrderReturnApplicationService;
import com.metawebthree.order.domain.model.OrderReturnApply;
import com.metawebthree.order.domain.model.ReturnApplyDO;
import com.metawebthree.order.infrastructure.persistence.mapper.ReturnApplyMapper;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Validated
@RestController
@RequestMapping("/returnApply")
@RequiredArgsConstructor
@Tag(name = "Order Return Management")
public class OrderReturnController {

    private final OrderReturnApplicationService returnService;
    private final ReturnApplyMapper returnApplyMapper;

    @Operation(summary = "Submit return application")
    @PostMapping
    public ApiResponse<Void> apply(@RequestBody OrderReturnApply apply) {
        returnService.applyReturn(apply);
        return ApiResponse.success();
    }

    @Operation(summary = "Get return application by id")
    @GetMapping("/{id}")
    public ApiResponse<OrderReturnApply> details(@PathVariable Long id) {
        return ApiResponse.success(returnService.getReturn(id));
    }

    @Operation(summary = "Update return application status")
    @PutMapping("/{id}")
    public ApiResponse<Void> handle(@PathVariable Long id, @RequestBody OrderReturnApply apply) {
        returnService.handleReturn(apply);
        return ApiResponse.success();
    }

    @Operation(summary = "List return applications by order Sn")
    @GetMapping("/order/{orderSn}")
    public ApiResponse<List<OrderReturnApply>> listByOrder(@PathVariable String orderSn) {
        return ApiResponse.success(returnService.listByOrder(orderSn));
    }

    @Operation(summary = "Remove application history")
    @DeleteMapping("/{id}")
    public ApiResponse<Void> remove(@PathVariable Long id) {
        returnService.removeHistory(id);
        return ApiResponse.success();
    }

    // === Admin endpoints ===

    @Operation(summary = "分页查询退货申请")
    @GetMapping("/list")
    public ApiResponse<Page<ReturnApplyDO>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) Long id,
            @RequestParam(required = false) Long orderId,
            @RequestParam(required = false) String orderSn,
            @RequestParam(required = false) Integer status) {
        LambdaQueryWrapper<ReturnApplyDO> wrapper = new LambdaQueryWrapper<ReturnApplyDO>()
                .orderByDesc(ReturnApplyDO::getId);
        if (id != null) wrapper.eq(ReturnApplyDO::getId, id);
        if (orderId != null) wrapper.eq(ReturnApplyDO::getOrderId, orderId);
        if (orderSn != null && !orderSn.isEmpty()) wrapper.eq(ReturnApplyDO::getOrderSn, orderSn);
        if (status != null) wrapper.eq(ReturnApplyDO::getStatus, status);
        return ApiResponse.success(returnApplyMapper.selectPage(new Page<>(pageNum, pageSize), wrapper));
    }

    @Operation(summary = "批量删除退货申请")
    @PostMapping("/delete")
    public ApiResponse<Void> delete(@RequestParam String ids) {
        List<Long> idList = Arrays.stream(ids.split(","))
                .map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        returnApplyMapper.deleteByIds(idList);
        return ApiResponse.success();
    }

    @Operation(summary = "修改退货申请状态")
    @PostMapping("/update/status/{id}")
    public ApiResponse<Void> updateStatus(@PathVariable Long id, @RequestBody Map<String, Object> params) {
        ReturnApplyDO entity = returnApplyMapper.selectById(id);
        if (entity == null) return ApiResponse.success();
        if (params.containsKey("status")) entity.setStatus(Integer.valueOf(String.valueOf(params.get("status"))));
        if (params.containsKey("companyAddressId"))
            entity.setCompanyAddressId(Long.valueOf(String.valueOf(params.get("companyAddressId"))));
        if (params.containsKey("returnAmount"))
            entity.setReturnAmount(new java.math.BigDecimal(String.valueOf(params.get("returnAmount"))));
        if (params.containsKey("handleNote")) entity.setHandleNote((String) params.get("handleNote"));
        if (params.containsKey("handleMan")) entity.setHandleMan((String) params.get("handleMan"));
        if (params.containsKey("receiveNote")) entity.setReceiveNote((String) params.get("receiveNote"));
        if (params.containsKey("receiveMan")) entity.setReceiveMan((String) params.get("receiveMan"));
        entity.setHandleTime(java.time.LocalDateTime.now());
        returnApplyMapper.updateById(entity);
        return ApiResponse.success();
    }
}
