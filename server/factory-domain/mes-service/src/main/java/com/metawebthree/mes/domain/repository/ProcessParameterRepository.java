package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.ProcessParameter;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

/**
 * 工艺参数仓储接口
 */
@Repository
public interface ProcessParameterRepository extends JpaRepository<ProcessParameter, Long> {
    
    /**
     * 根据参数编码查询
     */
    ProcessParameter findByParamCode(String paramCode);
    
    /**
     * 根据工艺路线ID查询参数列表
     */
    List<ProcessParameter> findByRouteIdOrderByStepNoAscDisplayOrderAsc(Long routeId);
    
    /**
     * 根据工艺路线编码查询参数列表
     */
    List<ProcessParameter> findByRouteCodeOrderByStepNoAscDisplayOrderAsc(String routeCode);
    
    /**
     * 根据工序查询参数列表
     */
    List<ProcessParameter> findByRouteIdAndStepNoOrderByDisplayOrderAsc(Long routeId, Integer stepNo);
    
    /**
     * 根据参数类型查询
     */
    List<ProcessParameter> findByParamType(ProcessParameter.ParamType paramType);
    
    /**
     * 根据状态查询参数列表
     */
    List<ProcessParameter> findByStatus(ProcessParameter.ParamStatus status);
    
    /**
     * 根据参数分组查询
     */
    List<ProcessParameter> findByParamGroup(String paramGroup);
    
    /**
     * 检查参数编码是否存在
     */
    boolean existsByParamCode(String paramCode);
    
    /**
     * 根据工艺路线ID统计参数数量
     */
    long countByRouteId(Long routeId);
    
    /**
     * 根据工序统计参数数量
     */
    long countByRouteIdAndStepNo(Long routeId, Integer stepNo);
    
    /**
     * 查询活跃状态的参数
     */
    @Query("SELECT p FROM ProcessParameter p WHERE p.routeId = :routeId AND p.status = 'ACTIVE' ORDER BY p.stepNo, p.displayOrder")
    List<ProcessParameter> findActiveByRouteId(@Param("routeId") Long routeId);
    
    /**
     * 根据参数类型和状态查询
     */
    List<ProcessParameter> findByParamTypeAndStatus(ProcessParameter.ParamType paramType, ProcessParameter.ParamStatus status);
}