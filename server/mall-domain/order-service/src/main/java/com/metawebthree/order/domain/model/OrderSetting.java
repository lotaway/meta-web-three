package com.metawebthree.order.domain.model;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
@Schema(description = "订单设置")
public class OrderSetting {
    @Schema(description = "设置ID")
    private final Long id;
    @Schema(description = "秒杀订单超时时间(分钟)")
    private final Integer flashOrderOvertime;
    @Schema(description = "普通订单超时时间(分钟)")
    private final Integer normalOrderOvertime;
    @Schema(description = "确认收货超时时间(天)")
    private final Integer confirmOvertime;
    @Schema(description = "完成订单超时时间(天)")
    private final Integer finishOvertime;
    @Schema(description = "评论超时时间(天)")
    private final Integer commentOvertime;
}
