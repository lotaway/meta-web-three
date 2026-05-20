package com.metawebthree.product.domain.model;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
@Schema(description = "商品属性")
public class Attribute {
    @Schema(description = "属性ID")
    private final Long id;
    @Schema(description = "分类ID")
    private final Long categoryId;
    @Schema(description = "属性名称")
    private final String name;
    @Schema(description = "选择类型")
    private final Integer selectType;
    @Schema(description = "输入类型")
    private final Integer inputType;
    @Schema(description = "可选值")
    private final String options;
    @Schema(description = "排序")
    private final Integer sort;
    @Schema(description = "筛选类型")
    private final Integer filterType;
    @Schema(description = "搜索类型")
    private final Integer searchType;
    @Schema(description = "是否关联")
    private final boolean related;
    @Schema(description = "是否支持手动添加")
    private final boolean supportsHandAdding;
    @Schema(description = "属性类型")
    private final Integer attributeType;

    public boolean isSpecification() {
        return attributeType != null && attributeType == 0;
    }

    public boolean isParameter() {
        return attributeType != null && attributeType == 1;
    }
}
