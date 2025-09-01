package com.metawebthree.media;

import com.github.yulichang.base.MPJBaseMapper;
import com.metawebthree.media.DO.ArtWorkTagDO;

import java.util.Collection;
import java.util.List;

import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface ArtWorkTagMapper extends MPJBaseMapper<ArtWorkTagDO> {
    @Insert({
        "<script>",
        "INSERT INTO Artwork_Tag (name) VALUES",
        "<foreach collection='tagNames' item='tag' separator=','>",
        "(#{tag})",
        "</foreach>",
        "RETURNING id",
        "</script>"
    })
    List<Integer> insertBatchThenReturnIds(@Param("tagNames") Collection<String> tagNames);
}