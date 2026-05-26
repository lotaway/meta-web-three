package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.ProcessParameter;

import java.util.List;
import java.util.Optional;

public interface ProcessParameterRepository {
    
    Optional<ProcessParameter> findById(Long id);
    
    Optional<ProcessParameter> findByParamCode(String paramCode);
    
    List<ProcessParameter> findByRouteIdOrderByStepNoAscDisplayOrderAsc(Long routeId);
    
    List<ProcessParameter> findByRouteCodeOrderByStepNoAscDisplayOrderAsc(String routeCode);
    
    List<ProcessParameter> findByRouteIdAndStepNoOrderByDisplayOrderAsc(Long routeId, Integer stepNo);
    
    List<ProcessParameter> findByParamType(ProcessParameter.ParamType paramType);
    
    List<ProcessParameter> findByStatus(ProcessParameter.ParamStatus status);
    
    List<ProcessParameter> findByParamGroup(String paramGroup);
    
    boolean existsByParamCode(String paramCode);
    
    long countByRouteId(Long routeId);
    
    long countByRouteIdAndStepNo(Long routeId, Integer stepNo);
    
    List<ProcessParameter> findActiveByRouteId(Long routeId);
    
    List<ProcessParameter> findByParamTypeAndStatus(ProcessParameter.ParamType paramType, ProcessParameter.ParamStatus status);
    
    ProcessParameter save(ProcessParameter parameter);
    
    List<ProcessParameter> saveAll(List<ProcessParameter> parameters);
    
    void deleteById(Long id);
    
    void deleteAllById(List<Long> ids);
    
    boolean existsById(Long id);
}