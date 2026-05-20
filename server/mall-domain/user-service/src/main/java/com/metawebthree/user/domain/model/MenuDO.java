package com.metawebthree.user.domain.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("tb_menu")
public class MenuDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private Long parentId;
    private LocalDateTime createTime;
    private String title;
    private Integer level;
    private Integer sort;
    private String name;
    private String icon;
    private Integer hidden;
}
