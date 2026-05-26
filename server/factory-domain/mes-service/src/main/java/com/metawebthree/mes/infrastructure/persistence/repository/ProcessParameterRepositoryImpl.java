package com.metawebthree.mes.infrastructure.persistence.repository;

import com.metawebthree.mes.domain.entity.ProcessParameter;
import com.metawebthree.mes.domain.repository.ProcessParameterRepository;
import org.springframework.stereotype.Repository;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

/**
 * 工艺参数仓储实现
 * 使用内存存储，生产环境应替换为数据库实现
 */
@Repository
public class ProcessParameterRepositoryImpl implements ProcessParameterRepository {
    
    private final Map<Long, ProcessParameter> storage = new ConcurrentHashMap<>();
    private final AtomicLong idGen = new AtomicLong(1);
    
    @Override
    public ProcessParameter findById(Long id) {
        return storage.get(id);
    }
    
    @Override
    public ProcessParameter findByParamCode(String paramCode) {
        return storage.values().stream()
                .filter(p -> p.getParamCode().equals(paramCode))
                .findFirst()
                .orElse(null);
    }
    
    @Override
    public List<ProcessParameter> findByRouteIdOrderByStepNoAscDisplayOrderAsc(Long routeId) {
        return storage.values().stream()
                .filter(p -> p.getRouteId().equals(routeId))
                .sorted(Comparator
                        .comparing(ProcessParameter::getStepNo, Comparator.nullsLast(Comparator.naturalOrder()))
                        .thenComparing(ProcessParameter::getDisplayOrder, Comparator.nullsLast(Comparator.naturalOrder())))
                .collect(Collectors.toList());
    }
    
    @Override
    public List<ProcessParameter> findByRouteCodeOrderByStepNoAscDisplayOrderAsc(String routeCode) {
        return storage.values().stream()
                .filter(p -> p.getRouteCode().equals(routeCode))
                .sorted(Comparator
                        .comparing(ProcessParameter::getStepNo, Comparator.nullsLast(Comparator.naturalOrder()))
                        .thenComparing(ProcessParameter::getDisplayOrder, Comparator.nullsLast(Comparator.naturalOrder())))
                .collect(Collectors.toList());
    }
    
    @Override
    public List<ProcessParameter> findByRouteIdAndStepNoOrderByDisplayOrderAsc(Long routeId, Integer stepNo) {
        return storage.values().stream()
                .filter(p -> p.getRouteId().equals(routeId))
                .filter(p -> p.getStepNo().equals(stepNo))
                .sorted(Comparator.comparing(ProcessParameter::getDisplayOrder, Comparator.nullsLast(Comparator.naturalOrder())))
                .collect(Collectors.toList());
    }
    
    @Override
    public List<ProcessParameter> findByParamType(ProcessParameter.ParamType paramType) {
        return storage.values().stream()
                .filter(p -> p.getParamType() == paramType)
                .collect(Collectors.toList());
    }
    
    @Override
    public List<ProcessParameter> findByStatus(ProcessParameter.ParamStatus status) {
        return storage.values().stream()
                .filter(p -> p.getStatus() == status)
                .collect(Collectors.toList());
    }
    
    @Override
    public List<ProcessParameter> findByParamGroup(String paramGroup) {
        return storage.values().stream()
                .filter(p -> p.getParamGroup() != null && p.getParamGroup().equals(paramGroup))
                .collect(Collectors.toList());
    }
    
    @Override
    public boolean existsByParamCode(String paramCode) {
        return storage.values().stream()
                .anyMatch(p -> p.getParamCode().equals(paramCode));
    }
    
    @Override
    public long countByRouteId(Long routeId) {
        return storage.values().stream()
                .filter(p -> p.getRouteId().equals(routeId))
                .count();
    }
    
    @Override
    public long countByRouteIdAndStepNo(Long routeId, Integer stepNo) {
        return storage.values().stream()
                .filter(p -> p.getRouteId().equals(routeId))
                .filter(p -> p.getStepNo().equals(stepNo))
                .count();
    }
    
    @Override
    public List<ProcessParameter> findActiveByRouteId(Long routeId) {
        return storage.values().stream()
                .filter(p -> p.getRouteId().equals(routeId))
                .filter(p -> p.getStatus() == ProcessParameter.ParamStatus.ACTIVE)
                .sorted(Comparator
                        .comparing(ProcessParameter::getStepNo, Comparator.nullsLast(Comparator.naturalOrder()))
                        .thenComparing(ProcessParameter::getDisplayOrder, Comparator.nullsLast(Comparator.naturalOrder())))
                .collect(Collectors.toList());
    }
    
    @Override
    public List<ProcessParameter> findByParamTypeAndStatus(ProcessParameter.ParamType paramType, ProcessParameter.ParamStatus status) {
        return storage.values().stream()
                .filter(p -> p.getParamType() == paramType)
                .filter(p -> p.getStatus() == status)
                .collect(Collectors.toList());
    }
    
    @Override
    public ProcessParameter save(ProcessParameter parameter) {
        if (parameter.getId() == null) {
            parameter.setId(idGen.getAndIncrement());
        }
        storage.put(parameter.getId(), parameter);
        return parameter;
    }
    
    @Override
    public List<ProcessParameter> saveAll(List<ProcessParameter> parameters) {
        List<ProcessParameter> saved = new ArrayList<>();
        for (ProcessParameter parameter : parameters) {
            saved.add(save(parameter));
        }
        return saved;
    }
    
    @Override
    public void deleteById(Long id) {
        storage.remove(id);
    }
    
    @Override
    public void deleteAllById(List<Long> ids) {
        for (Long id : ids) {
            storage.remove(id);
        }
    }
    
    @Override
    public boolean existsById(Long id) {
        return storage.containsKey(id);
    }
}