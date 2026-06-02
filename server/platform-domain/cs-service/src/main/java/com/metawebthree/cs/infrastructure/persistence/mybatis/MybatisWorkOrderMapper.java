package com.metawebthree.cs.infrastructure.persistence.mybatis;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.cs.domain.model.WorkOrder;
import com.metawebthree.cs.domain.model.enums.WorkOrderCategory;
import com.metawebthree.cs.domain.model.enums.WorkOrderStatus;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface MybatisWorkOrderMapper extends BaseMapper<WorkOrder> {

    @Select("SELECT * FROM cs_work_order WHERE customer_id = #{customerId} ORDER BY create_time DESC")
    List<WorkOrder> findByCustomerId(@Param("customerId") Long customerId);

    @Select("SELECT * FROM cs_work_order WHERE agent_id = #{agentId} ORDER BY create_time DESC")
    List<WorkOrder> findByAgentId(@Param("agentId") Long agentId);

    @Select("SELECT * FROM cs_work_order WHERE status = #{status} ORDER BY priority DESC, create_time ASC")
    List<WorkOrder> findByStatus(@Param("status") WorkOrderStatus status);

    @Select("SELECT * FROM cs_work_order WHERE category = #{category} ORDER BY create_time DESC")
    List<WorkOrder> findByCategory(@Param("category") WorkOrderCategory category);

    @Select("SELECT * FROM cs_work_order WHERE status IN ('PENDING', 'PROCESSING') ORDER BY priority DESC, create_time ASC")
    List<WorkOrder> findPending();

    @Select("SELECT COUNT(*) FROM cs_work_order WHERE status = #{status}")
    Long countByStatus(@Param("status") WorkOrderStatus status);

    @Select("SELECT COUNT(*) FROM cs_work_order WHERE category = #{category}")
    Long countByCategory(@Param("category") WorkOrderCategory category);

    @Update("UPDATE cs_work_order SET status = #{status}, update_time = NOW() WHERE id = #{id}")
    void updateStatus(@Param("id") Long id, @Param("status") WorkOrderStatus status);

    @Update("UPDATE cs_work_order SET resolution = #{resolution}, status = 'RESOLVED', resolve_time = NOW(), update_time = NOW() WHERE id = #{id}")
    void resolve(@Param("id") Long id, @Param("resolution") String resolution);

    @Update("UPDATE cs_work_order SET agent_id = #{agentId}, status = 'PROCESSING', update_time = NOW() WHERE id = #{id}")
    void assign(@Param("id") Long id, @Param("agentId") Long agentId);
}