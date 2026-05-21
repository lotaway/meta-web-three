package com.metawebthree.action.domain.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@TableName("tb_product_comment")
public class ProductComment {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long productId;
    private Long userId;
    private String memberNickName;
    private String productName;
    private Integer star;
    private String content;
    private String pics;
    private String productAttribute;
    private Integer showStatus;
    private Integer collectCount;
    private Integer readCount;
    private Integer replayCount;
    private LocalDateTime createTime;
}
