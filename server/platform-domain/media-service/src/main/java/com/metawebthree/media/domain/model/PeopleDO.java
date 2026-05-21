package com.metawebthree.media.domain.model;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.metawebthree.common.DO.BaseDO;
import com.metawebthree.common.adapter.SQLLongArrayHandler;

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
    @TableId(type = IdType.ASSIGN_ID)
    Long id;
    String name;
    @TableField(typeHandler = SQLLongArrayHandler.class)
    Long[] types;
}
