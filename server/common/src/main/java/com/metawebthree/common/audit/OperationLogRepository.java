package com.metawebthree.common.audit;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.time.LocalDateTime;
import java.util.List;

@Mapper
public interface OperationLogRepository extends BaseMapper<OperationLog> {

    @Select("SELECT * FROM tb_operation_log WHERE user_id = #{userId} ORDER BY operation_time DESC")
    List<OperationLog> findByUserIdOrderByOperationTimeDesc(@Param("userId") Long userId);

    @Select("SELECT * FROM tb_operation_log WHERE operation = #{operation} ORDER BY operation_time DESC")
    List<OperationLog> findByOperationOrderByOperationTimeDesc(@Param("operation") String operation);

    @Select("SELECT * FROM tb_operation_log WHERE status = #{status} ORDER BY operation_time DESC")
    List<OperationLog> findByStatusOrderByOperationTimeDesc(@Param("status") String status);

    @Select("SELECT * FROM tb_operation_log WHERE entity_type = #{entityType} AND entity_id = #{entityId} ORDER BY operation_time DESC")
    List<OperationLog> findByEntityTypeAndEntityIdOrderByOperationTimeDesc(
            @Param("entityType") String entityType,
            @Param("entityId") Long entityId);

    @Select("SELECT * FROM tb_operation_log WHERE operation_time BETWEEN #{startTime} AND #{endTime} ORDER BY operation_time DESC")
    List<OperationLog> findByOperationTimeBetweenOrderByOperationTimeDesc(
            @Param("startTime") LocalDateTime startTime,
            @Param("endTime") LocalDateTime endTime);

    @Select("SELECT COUNT(*) FROM tb_operation_log WHERE user_id = #{userId}")
    long countByUserId(@Param("userId") Long userId);

    @Select("SELECT COUNT(*) FROM tb_operation_log WHERE status = #{status}")
    long countByStatus(@Param("status") String status);

    @Select("SELECT * FROM tb_operation_log WHERE status = 'FAILURE' OR status = 'ERROR' ORDER BY operation_time DESC")
    List<OperationLog> findFailedOperations();
}
