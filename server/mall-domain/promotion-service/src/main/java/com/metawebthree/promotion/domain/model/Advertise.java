package com.metawebthree.promotion.domain.model;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Getter;
import java.time.LocalDateTime;

@Getter
@Builder
@Schema(description = "首页广告")
public class Advertise {
    @Schema(description = "广告ID")
    private final Long id;
    @Schema(description = "广告名称")
    private final String name;
    @Schema(description = "轮播位置：0->PC首页轮播；1->APP首页轮播")
    private final Integer type;
    @Schema(description = "图片地址")
    private final String pic;
    @Schema(description = "开始时间")
    private final LocalDateTime startTime;
    @Schema(description = "结束时间")
    private final LocalDateTime endTime;
    @Schema(description = "上下线状态：0->下线；1->上线")
    private final Integer status;
    @Schema(description = "点击数")
    private final Integer clickCount;
    @Schema(description = "下单数")
    private final Integer orderCount;
    @Schema(description = "链接地址")
    private final String url;
    @Schema(description = "备注")
    private final String note;
    @Schema(description = "排序")
    private final Integer sort;
}
