package com.metawebthree.media.DO;

import com.baomidou.mybatisplus.annotation.TableName;
import com.metawebthree.common.DO.BaseDO;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Data
@NoArgsConstructor
@AllArgsConstructor
@SuperBuilder
@TableName("People_Type")
public class PeopleTypeDO extends BaseDO {
    Short id;
    String type;
}
