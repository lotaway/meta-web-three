package com.metawebthree.product.domain.model;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
@AllArgsConstructor
@Schema(description = "商品分类")
public class ProductCategory {
    @Schema(description = "分类ID")
    private final Long id;
    @Schema(description = "父分类ID")
    private final Long parentId;
    @Schema(description = "分类名称")
    private final String name;
    @Schema(description = "层级")
    private final Integer level;
    @Schema(description = "商品数量")
    private final Integer productCount;
    @Schema(description = "商品单位")
    private final String productUnit;
    @Schema(description = "是否显示在导航栏")
    private final boolean displayedInNav;
    @Schema(description = "是否可见")
    private final boolean visible;
    @Schema(description = "排序")
    private final Integer sort;
    @Schema(description = "图标")
    private final String icon;
    @Schema(description = "关键词")
    private final String keywords;
    @Schema(description = "描述")
    private final String description;

    public boolean isRoot() {
        return parentId == null || parentId == 0;
    }
}
