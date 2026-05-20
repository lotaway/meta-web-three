package com.metawebthree.promotion.infrastructure.persistence.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("tb_cms_subject")
public class CmsSubjectDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private Long categoryId;
    private String title;
    private String pic;
    private Integer productCount;
    private Integer recommendStatus;
    private LocalDateTime createTime;
    private Integer collectCount;
    private Integer readCount;
    private Integer commentCount;
    private String albumPics;
    private String description;
    private Integer showStatus;
    private Integer forwardCount;
    private String categoryName;
    private String content;
}
