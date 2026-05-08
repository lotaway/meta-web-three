package com.metawebthree.promotion.interfaces.web;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.promotion.infrastructure.persistence.mapper.FlashPromotionSessionMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.FlashPromotionSessionDO;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@Validated
@RestController
@RequestMapping("/flashSession")
@RequiredArgsConstructor
@Tag(name = "Flash Promotion Session Admin")
public class FlashPromotionSessionController {

    private final FlashPromotionSessionMapper sessionMapper;

    @Operation(summary = "获取全部场次")
    @GetMapping("/list")
    public ApiResponse<List<FlashPromotionSessionDO>> list() {
        return ApiResponse.success(sessionMapper.selectList(null));
    }

    @Operation(summary = "获取全部可选场次及其数量")
    @GetMapping("/selectList")
    public ApiResponse<List<FlashPromotionSessionDO>> selectList(@RequestParam(required = false) Long flashPromotionId) {
        return ApiResponse.success(sessionMapper.selectList(null));
    }

    @Operation(summary = "修改场次启用状态")
    @PostMapping("/update/status/{id}")
    public ApiResponse<Void> updateStatus(@PathVariable Long id, @RequestParam Integer status) {
        sessionMapper.update(null, new UpdateWrapper<FlashPromotionSessionDO>().eq("id", id).set("status", status));
        return ApiResponse.success();
    }

    @Operation(summary = "删除场次")
    @PostMapping("/delete/{id}")
    public ApiResponse<Void> delete(@PathVariable Long id) {
        sessionMapper.deleteById(id);
        return ApiResponse.success();
    }

    @Operation(summary = "添加场次")
    @PostMapping("/create")
    public ApiResponse<Void> create(@RequestBody FlashPromotionSessionDO session) {
        sessionMapper.insert(session);
        return ApiResponse.success();
    }

    @Operation(summary = "修改场次")
    @PostMapping("/update/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody FlashPromotionSessionDO session) {
        session.setId(id);
        sessionMapper.updateById(session);
        return ApiResponse.success();
    }
}
