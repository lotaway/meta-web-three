package com.meta.common.audit;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.time.LocalDateTime;
import java.util.List;

@Mapper
public interface AuditLogRepository extends BaseMapper<AuditLog> {

    @Select("SELECT * FROM tb_audit_log WHERE username = #{username} ORDER BY operation_time DESC")
    List<AuditLog> findByUsernameOrderByOperationTimeDesc(@Param("username") String username);

    @Select("SELECT * FROM tb_audit_log WHERE operation_type = #{operationType} ORDER BY operation_time DESC")
    List<AuditLog> findByOperationTypeOrderByOperationTimeDesc(@Param("operationType") String operationType);

    @Select("SELECT * FROM tb_audit_log WHERE resource_type = #{resourceType} ORDER BY operation_time DESC")
    List<AuditLog> findByResourceTypeOrderByOperationTimeDesc(@Param("resourceType") String resourceType);

    @Select("SELECT * FROM tb_audit_log WHERE operation_time BETWEEN #{startTime} AND #{endTime} ORDER BY operation_time DESC")
    List<AuditLog> findByOperationTimeBetweenOrderByOperationTimeDesc(
            @Param("startTime") LocalDateTime startTime,
            @Param("endTime") LocalDateTime endTime);

    @Select("DELETE FROM tb_audit_log WHERE operation_time < #{time}")
    int deleteBefore(@Param("time") LocalDateTime time);
}
