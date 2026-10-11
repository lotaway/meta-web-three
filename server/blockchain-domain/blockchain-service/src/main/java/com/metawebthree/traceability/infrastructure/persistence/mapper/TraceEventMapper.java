package com.metawebthree.traceability.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.traceability.domain.entity.TraceEventDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface TraceEventMapper extends BaseMapper<TraceEventDO> {

    @Select("SELECT * FROM trace_event WHERE trace_id = #{traceId} ORDER BY timestamp ASC")
    List<TraceEventDO> selectByTraceId(Long traceId);

    @Select("SELECT COUNT(*) FROM trace_event WHERE trace_id = #{traceId}")
    Integer selectCountByTraceId(Long traceId);
}