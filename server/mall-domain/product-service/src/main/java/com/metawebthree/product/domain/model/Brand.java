package com.metawebthree.product.domain.model;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
@Schema(description = "品牌")
public class Brand {
    @Schema(description = "品牌ID")
    private final Long id;
    @Schema(description = "品牌名称")
    private final String name;
    @Schema(description = "首字母")
    private final String firstLetter;
    @Schema(description = "排序")
    private final Integer sort;
    @Schema(description = "是否为厂商")
    private final boolean manufacturer;
    @Schema(description = "是否可见")
    private final boolean visible;
    @Schema(description = "商品数量")
    private final Integer productCount;
    @Schema(description = "评论数量")
    private final Integer commentCount;
    @Schema(description = "Logo")
    private final String logo;
    @Schema(description = "区域图片")
    private final String areaPicture;
    @Schema(description = "品牌故事")
    private final String story;
}
