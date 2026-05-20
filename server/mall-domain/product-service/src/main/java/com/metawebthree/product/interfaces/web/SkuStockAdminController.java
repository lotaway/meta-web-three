package com.metawebthree.product.interfaces.web;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.product.domain.model.SkuStockDO;
import com.metawebthree.product.infrastructure.persistence.mapper.SkuStockMapper;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@Validated
@RestController
@RequestMapping("/sku")
@RequiredArgsConstructor
@Tag(name = "Sku Stock Admin")
public class SkuStockAdminController {

    private final SkuStockMapper skuStockMapper;

    @Operation(summary = "根据商品ID及sku编码模糊搜索sku库存")
    @GetMapping("/{pid}")
    public ApiResponse<List<SkuStockDO>> list(@PathVariable Long pid,
                                              @RequestParam(required = false) String keyword) {
        LambdaQueryWrapper<SkuStockDO> wrapper = new LambdaQueryWrapper<SkuStockDO>()
                .eq(SkuStockDO::getProductId, pid);
        if (keyword != null && !keyword.isEmpty()) {
            wrapper.like(SkuStockDO::getSkuCode, keyword);
        }
        return ApiResponse.success(skuStockMapper.selectList(wrapper));
    }

    @Operation(summary = "根据商品ID批量更新sku库存信息")
    @PostMapping("/update/{pid}")
    public ApiResponse<Void> update(@PathVariable Long pid, @RequestBody List<SkuStockDO> skuList) {
        for (SkuStockDO sku : skuList) {
            sku.setProductId(pid);
            if (sku.getId() != null && sku.getId() > 0) {
                skuStockMapper.updateById(sku);
            } else {
                skuStockMapper.insert(sku);
            }
        }
        return ApiResponse.success();
    }
}
