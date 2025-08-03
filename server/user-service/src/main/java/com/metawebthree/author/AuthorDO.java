package com.metawebthree.author;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@TableName("Author")
public class AuthorDO {
    private Integer id;
    @TableField("real_name")
    private String realName;
    @TableField("is_enable")
    private Boolean isEnable;
}
