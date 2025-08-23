package com.metawebthree.media.DO;

import java.sql.Array;
import java.util.List;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import com.metawebthree.common.DO.BaseDO;
import com.metawebthree.common.adapter.IntegerArrayTypeHandler;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Data
@NoArgsConstructor
@AllArgsConstructor
@SuperBuilder
@TableName("\"Artwork\"")
public class ArtWorkDO extends BaseDO {
    Long id;
    String series;
    String title;
    String cover;
    String link;
    String subtitle;
    Integer season;
    Integer episode;
    Integer categoryId;
    @TableField(typeHandler = IntegerArrayTypeHandler.class)
    Integer[] tags;
    Integer yearTag;
    @TableField(typeHandler = IntegerArrayTypeHandler.class)
    Integer[] acts;
    Integer director;
}
