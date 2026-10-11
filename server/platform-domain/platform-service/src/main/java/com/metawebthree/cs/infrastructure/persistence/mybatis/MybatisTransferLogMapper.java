package com.metawebthree.cs.infrastructure.persistence.mybatis;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.cs.domain.model.TransferLog;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface MybatisTransferLogMapper extends BaseMapper<TransferLog> {

    @Insert("INSERT INTO cs_transfer_log (session_id, from_agent_id, to_agent_id, reason, transfer_time) " +
            "VALUES (#{sessionId}, #{fromAgentId}, #{toAgentId}, #{reason}, #{transferTime})")
    @Options(useGeneratedKeys = true, keyProperty = "id")
    int insert(TransferLog transferLog);

    @Select("SELECT * FROM cs_transfer_log WHERE session_id = #{sessionId} ORDER BY transfer_time DESC")
    List<TransferLog> findBySessionId(@Param("sessionId") String sessionId);
}