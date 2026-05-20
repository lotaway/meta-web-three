package com.metawebthree.promotion.infrastructure.persistence.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("tb_coupon")
public class CouponDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private Integer type;
    private String name;
    private Integer platform;
    private Integer count;
    private BigDecimal amount;
    private Integer perLimit;
    private BigDecimal minPoint;
    private LocalDateTime startTime;
    private LocalDateTime endTime;
    private Integer useType;
    private String note;
    private Integer publishCount;
    private Integer useCount;
    private Integer receiveCount;
    private LocalDateTime enableTime;
    private String code;
    private Integer memberLevel;
}
