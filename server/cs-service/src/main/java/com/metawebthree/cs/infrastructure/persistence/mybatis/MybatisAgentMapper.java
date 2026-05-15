package com.metawebthree.cs.infrastructure.persistence.mybatis;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.cs.domain.model.Agent;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Update;

@Mapper
public interface MybatisAgentMapper extends BaseMapper<Agent> {
    @Update("UPDATE cs_agent SET status = #{status}, update_time = NOW() WHERE id = #{id}")
    void updateStatus(@Param("id") Long id, @Param("status") String status);

    @Update("UPDATE cs_agent SET current_load = current_load + #{delta}, update_time = NOW() WHERE id = #{id}")
    void updateLoad(@Param("id") Long id, @Param("delta") int delta);
}
