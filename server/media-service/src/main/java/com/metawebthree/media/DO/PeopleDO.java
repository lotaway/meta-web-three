package com.metawebthree.media.DO;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import com.metawebthree.common.DO.BaseDO;
import com.metawebthree.common.adapter.ShortArrayTypeHandler;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Data
@NoArgsConstructor
@AllArgsConstructor
@SuperBuilder
@TableName("\"People\"")
public class PeopleDO extends BaseDO {
    Integer id;
    String name;
    @TableField(typeHandler = ShortArrayTypeHandler.class)
    Short[] types;
}
