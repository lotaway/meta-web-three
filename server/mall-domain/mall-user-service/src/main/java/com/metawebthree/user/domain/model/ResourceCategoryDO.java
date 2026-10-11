package com.metawebthree.user.domain.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("tb_resource_category")
public class ResourceCategoryDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private LocalDateTime createTime;
    private String name;
    private Integer sort;
}
