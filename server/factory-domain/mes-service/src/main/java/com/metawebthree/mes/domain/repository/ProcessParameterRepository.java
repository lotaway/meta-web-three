package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.ProcessParameter;

import java.util.List;

/**
 * 工艺参数仓储接口
 */
public interface ProcessParameterRepository {
    
    /**
     * 根据ID查询
     */
    ProcessParameter findById(Long id);
    
    /**
     * 根据参数编码查询
     */
    ProcessParameter findByParamCode(String paramCode);
    
    /**
     * 根据工艺路线ID查询参数列表（按工序序号和显示顺序排序）
     */
    List<ProcessParameter> findByRouteIdOrderByStepNoAscDisplayOrderAsc(Long routeId);
    
    /**
     * 根据工艺路线编码查询参数列表（按工序序号和显示顺序排序）
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
     * 查询激活状态的参数（按工序序号和显示顺序排序）
     */
    List<ProcessParameter> findActiveByRouteId(Long routeId);
    
    /**
     * 根据参数类型和状态查询
     */
    List<ProcessParameter> findByParamTypeAndStatus(ProcessParameter.ParamType paramType, ProcessParameter.ParamStatus status);
    
    /**
     * 保存工艺参数
     */
    ProcessParameter save(ProcessParameter parameter);
    
    /**
     * 批量保存工艺参数
     */
    List<ProcessParameter> saveAll(List<ProcessParameter> parameters);
    
    /**
     * 根据ID删除
     */
    void deleteById(Long id);
    
    /**
     * 批量删除
     */
    void deleteAllById(List<Long> ids);
    
    /**
     * 检查ID是否存在
     */
    boolean existsById(Long id);
}