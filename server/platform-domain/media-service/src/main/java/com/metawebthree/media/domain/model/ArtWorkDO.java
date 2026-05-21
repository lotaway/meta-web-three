package com.metawebthree.media.domain.model;

import java.sql.Array;
import java.util.List;

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
@TableName("\"Artwork\"")
public class ArtWorkDO extends BaseDO implements Cloneable {
    @TableId(type = IdType.ASSIGN_ID)
    Long id;
    String series;
    String title;
    String cover;
    String link;
    String subtitle;
    Integer season;
    Integer episode;
    Long categoryId;
    @TableField(typeHandler = SQLLongArrayHandler.class)
    Long[] tags;
    Integer yearTag;
    @TableField(typeHandler = SQLLongArrayHandler.class)
    Long[] acts;
    Long director;

    @Override
    public ArtWorkDO clone() throws CloneNotSupportedException {
        BaseDO baseDO = super.clone();
        ArtWorkDO other = new ArtWorkDO();
        other.setId(this.getId());
        other.setSeries(this.getSeries());
        other.setTitle(this.getTitle());
        other.setCover(this.getCover());
        other.setLink(this.getLink());
        other.setSubtitle(this.getSubtitle());
        other.setSeason(this.getSeason());
        other.setEpisode(this.getEpisode());
        other.setCategoryId(this.getCategoryId());
        other.setTags(this.getTags());
        other.setYearTag(this.getYearTag());
        other.setActs(this.getActs());
        other.setDirector(this.getDirector());
        other.setCreatedAt(baseDO.getCreatedAt());
        other.setUpdatedAt(baseDO.getUpdatedAt());
        return other;
    }
}
