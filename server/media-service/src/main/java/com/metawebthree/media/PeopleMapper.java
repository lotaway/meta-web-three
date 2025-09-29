package com.metawebthree.media;

import com.github.yulichang.base.MPJBaseMapper;
import com.metawebthree.media.DO.PeopleDO;

import io.lettuce.core.dynamic.annotation.Param;

import java.util.Collection;
import java.util.List;

import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface PeopleMapper extends MPJBaseMapper<PeopleDO> {
    @Insert({
            "<script>",
            "INSERT INTO \"People\" (name,types)",
            "<foreach collection='list' item='item' separator=',' open='VALUES' close=''>",
            "(#{item.name},#{item.types})",
            "</foreach>",
            "RETURNING id",
            "</script>"
    })
    List<Integer> insertBatchThenReturnIds(@Param("list") Collection<PeopleDO> peopleDOs);
}
