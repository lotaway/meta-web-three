package com.metawebthree.traceability.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.traceability.domain.entity.TraceRecordDO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface TraceRecordMapper extends BaseMapper<TraceRecordDO> {

    @Select("SELECT * FROM trace_record WHERE trace_id = #{traceId}")
    TraceRecordDO selectByTraceId(Long traceId);

    @Select("SELECT * FROM trace_record WHERE product_id = #{productId}")
    List<TraceRecordDO> selectByProductId(String productId);

    @Select("SELECT MAX(trace_id) FROM trace_record")
    Long selectMaxTraceId();

    @Select("SELECT trace_id FROM trace_record WHERE product_id = #{productId}")
    List<Long> selectTraceIdsByProductId(String productId);
}