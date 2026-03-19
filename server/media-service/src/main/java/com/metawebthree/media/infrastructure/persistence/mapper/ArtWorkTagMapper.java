package com.metawebthree.media.infrastructure.persistence.mapper;

import com.github.yulichang.base.MPJBaseMapper;
import com.metawebthree.media.domain.model.ArtWorkTagDO;

import java.util.Collection;
import java.util.List;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

@Mapper
public interface ArtWorkTagMapper extends MPJBaseMapper<ArtWorkTagDO> {
    @Select({
        "<script>",
        "INSERT INTO \"Artwork_Tag\" (tag)",
        "<foreach collection='tagNames' item='tag' separator=',' open='VALUES' close=''>",
        "(#{tag})",
        "</foreach>",
        "RETURNING id",
        "</script>"
    })
    List<Integer> insertBatchThenReturnIds(@Param("tagNames") Collection<String> tagNames);
}