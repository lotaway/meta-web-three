package com.metawebthree.media;

import com.github.yulichang.base.MPJBaseMapper;
import com.metawebthree.media.DO.ArtWorkDO;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.cursor.Cursor;

@Mapper
public interface ArtWorkMapper extends MPJBaseMapper<ArtWorkDO> {
    @Select("SELECT id, series, title, cover, link, subtitle, season, episode, category_id, tags, year_tag, acts, director FROM Art_Work")
    Cursor<ArtWorkDO> getCursor();
}
