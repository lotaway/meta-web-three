package com.metawebthree.promotion.infrastructure.persistence.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
@TableName("tb_flash_promotion")
public class FlashPromotionDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private String title;
    private LocalDate startDate;
    private LocalDate endDate;
    private Integer status;
    private LocalDateTime createTime;
}
