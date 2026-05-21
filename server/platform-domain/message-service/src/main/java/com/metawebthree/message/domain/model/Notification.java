package com.metawebthree.message.domain.model;

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
@TableName("tb_notification")
public class Notification {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long userId;
    private String title;
    private String content;
    private String icon;
    private String imageUrl;
    private String type;
    private String relatedId;
    private Integer readStatus;
    private LocalDateTime createTime;
}
