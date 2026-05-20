package com.metawebthree.promotion.infrastructure.persistence.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;
import java.time.LocalTime;

@Data
@TableName("tb_flash_promotion_session")
public class FlashPromotionSessionDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private String name;
    private LocalTime startTime;
    private LocalTime endTime;
    private Integer status;
    private LocalDateTime createTime;
}
