package com.metawebthree.action.domain.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Builder;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.AllArgsConstructor;
import java.time.LocalDateTime;

@Data

@AllArgsConstructor
@Builder
@TableName("tb_member_brand_attention")
public class BrandAttention {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private Long userId;
    private Long brandId;
    private String brandName;
    private String brandLogo;
    private LocalDateTime createTime;
}
