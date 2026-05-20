package com.metawebthree.promotion.infrastructure.persistence.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("tb_cms_subject_category")
public class CmsSubjectCategoryDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private String name;
    private String icon;
    private Integer subjectCount;
    private Integer showStatus;
    private Integer sort;
}
