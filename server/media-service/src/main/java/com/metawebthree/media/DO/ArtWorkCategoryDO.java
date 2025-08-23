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
@TableName("Artwork_Category")
public class ArtWorkCategoryDO extends BaseDO {
    Long id;
    String name;
}
