package com.metawebthree.common.DO;

import java.sql.Timestamp;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.TableField;

public class BaseDO {
    @TableField(fill = FieldFill.INSERT)
    Timestamp createAt;
    @TableField(fill = FieldFill.INSERT_UPDATE)
    Timestamp updateAt;
}
