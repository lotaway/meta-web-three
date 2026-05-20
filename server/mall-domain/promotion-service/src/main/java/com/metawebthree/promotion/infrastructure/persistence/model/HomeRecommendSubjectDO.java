package com.metawebthree.promotion.infrastructure.persistence.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("tb_home_recommend_subject")
public class HomeRecommendSubjectDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private Long subjectId;
    private String subjectName;
    private Integer recommendStatus;
    private Integer sort;
}
