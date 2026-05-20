package com.metawebthree.payment.application.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;
import lombok.AllArgsConstructor;

import java.math.BigDecimal;
import java.time.LocalDateTime;

import com.metawebthree.common.DO.BaseDO;

@Data
@NoArgsConstructor
@AllArgsConstructor
@SuperBuilder
@Schema(description = "加密货币价格响应")
public class CryptoPriceResponse extends BaseDO {
    
    @Schema(description = "交易对符号")
    private String symbol;
    @Schema(description = "基础货币")
    private String baseCurrency;
    @Schema(description = "报价货币")
    private String quoteCurrency;
    @Schema(description = "当前价格")
    private BigDecimal price;
    @Schema(description = "买一价")
    private BigDecimal bidPrice;
    @Schema(description = "卖一价")
    private BigDecimal askPrice;
    @Schema(description = "24小时成交量")
    private BigDecimal volume24h;
    @Schema(description = "24小时涨跌")
    private BigDecimal change24h;
    @Schema(description = "24小时涨跌幅")
    private BigDecimal changePercent24h;
    @Schema(description = "数据来源")
    private String source;
    @Schema(description = "时间戳")
    private LocalDateTime timestamp;
} 