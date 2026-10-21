package com.metawebthree.mes.domain.service;

import com.metawebthree.mes.domain.config.StatusMachine;
import com.metawebthree.mes.domain.config.StatusConfig;
import com.metawebthree.mes.domain.config.StatusTransitionRule;
import com.metawebthree.mes.domain.repository.StatusMachineRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class StatusMachineService {
    
    private final StatusMachineRepository statusMachineRepository;
    
    public StatusMachineService(StatusMachineRepository statusMachineRepository) {
        this.statusMachineRepository = statusMachineRepository;
    }
    
    /**
     * 获取实体类型对应的状态机
     */
    public Optional<StatusMachine> getStatusMachine(String entityType) {
        return statusMachineRepository.findByEntityType(entityType);
    }
    
    /**
     * 根据状态机编码获取状态机
     */
    public Optional<StatusMachine> getStatusMachineByCode(String machineCode) {
        return statusMachineRepository.findByMachineCode(machineCode);
    }
    
    /**
     * 验证状态转换是否合法
     */
    public boolean isTransitionValid(StatusMachine machine, String fromStatus, String toStatus) {
        if (machine == null || machine.getTransitions() == null) {
            return false;
        }
        
        return machine.getTransitions().stream()
                .anyMatch(t -> t.getFromStatus().equals(fromStatus) 
                           && t.getToStatus().equals(toStatus));
    }
    
    /**
     * 获取状态转换规则
     */
    public Optional<StatusTransitionRule> getTransitionRule(
            StatusMachine machine, String fromStatus, String toStatus) {
        if (machine == null || machine.getTransitions() == null) {
            return Optional.empty();
        }
        
        return machine.getTransitions().stream()
                .filter(t -> t.getFromStatus().equals(fromStatus) 
                          && t.getToStatus().equals(toStatus))
                .findFirst();
    }
    
    /**
     * 获取指定状态的所有可转换目标状态
     */
    public List<String> getNextValidStatuses(StatusMachine machine, String currentStatus) {
        if (machine == null || machine.getTransitions() == null) {
            return List.of();
        }
        
        return machine.getTransitions().stream()
                .filter(t -> t.getFromStatus().equals(currentStatus))
                .map(StatusTransitionRule::getToStatus)
                .collect(java.util.stream.Collectors.toList());
    }
    
    /**
     * 获取状态配置信息
     */
    public Optional<StatusConfig> getStatusConfig(StatusMachine machine, String statusCode) {
        if (machine == null || machine.getStatuses() == null) {
            return Optional.empty();
        }
        
        return machine.getStatuses().stream()
                .filter(s -> s.getStatusCode().equals(statusCode))
                .findFirst();
    }
    
    /**
     * 检查是否为终态
     */
    public boolean isFinalStatus(StatusMachine machine, String statusCode) {
        return getStatusConfig(machine, statusCode)
                .map(StatusConfig::getIsFinal)
                .orElse(false);
    }
    
    /**
     * 获取初始状态
     */
    public Optional<String> getInitialStatus(StatusMachine machine) {
        if (machine == null || machine.getStatuses() == null) {
            return Optional.empty();
        }
        
        return machine.getStatuses().stream()
                .filter(StatusConfig::getIsInitial)
                .map(StatusConfig::getStatusCode)
                .findFirst();
    }
    
    /**
     * 校验实体从当前状态到目标状态的转换是否被状态机允许。
     * <p>
     * 状态机不存在时视为校验通过（未配置状态机的实体类型不做限制）；
     * 状态机存在但未定义该转换时抛出 {@link IllegalStateException}。
     * </p>
     *
     * @param entity    业务实体，需暴露 {@code getStatusCode()}
     * @param toStatus  目标状态码
     */
    public void validateTransition(Object entity, String toStatus) {
        if (entity == null) {
            throw new IllegalArgumentException("Entity must not be null");
        }
        String fromStatus = extractStatus(entity);
        StatusMachine machine = resolveMachine(entity);
        if (machine == null) {
            return;
        }
        if (!isTransitionValid(machine, fromStatus, toStatus)) {
            throw new IllegalStateException(
                    "Illegal status transition for " + machine.getMachineCode()
                    + ": " + fromStatus + " -> " + toStatus);
        }
    }

    /**
     * 校验指定实体类型的状态转换是否合法。
     *
     * @param entityType 实体类型
     * @param fromStatus 当前状态码
     * @param toStatus   目标状态码
     */
    public void validateTransition(String entityType, String fromStatus, String toStatus) {
        StatusMachine machine = getStatusMachine(entityType).orElse(null);
        if (machine == null) {
            return;
        }
        if (!isTransitionValid(machine, fromStatus, toStatus)) {
            throw new IllegalStateException(
                    "Illegal status transition for " + entityType + ": " + fromStatus + " -> " + toStatus);
        }
    }

    private StatusMachine resolveMachine(Object entity) {
        String machineCode = extractMachineCode(entity);
        if (machineCode != null) {
            StatusMachine byCode = getStatusMachineByCode(machineCode).orElse(null);
            if (byCode != null) {
                return byCode;
            }
        }
        String entityType = extractEntityType(entity);
        return entityType != null ? getStatusMachine(entityType).orElse(null) : null;
    }

    private String extractStatus(Object entity) {
        try {
            java.lang.reflect.Method getter = entity.getClass().getMethod("getStatusCode");
            Object value = getter.invoke(entity);
            return value != null ? value.toString() : null;
        } catch (ReflectiveOperationException e) {
            throw new IllegalArgumentException(
                    "Entity does not expose getStatusCode(): " + entity.getClass().getName(), e);
        }
    }

    /**
     * 读取实体声明的状态机编码（可选，通过静态常量或 {@code getStatusMachineCode()}）。
     */
    private String extractMachineCode(Object entity) {
        try {
            java.lang.reflect.Method getter = entity.getClass().getMethod("getStatusMachineCode");
            Object value = getter.invoke(entity);
            return value != null ? value.toString() : null;
        } catch (NoSuchMethodException e) {
            return null;
        } catch (ReflectiveOperationException e) {
            return null;
        }
    }

    private String extractEntityType(Object entity) {
        try {
            java.lang.reflect.Method getter = entity.getClass().getMethod("getEntityType");
            Object value = getter.invoke(entity);
            return value != null ? value.toString() : null;
        } catch (ReflectiveOperationException e) {
            return null;
        }
    }

    /**
     * 保存状态机配置
     */
    public StatusMachine saveStatusMachine(StatusMachine statusMachine) {
        return statusMachineRepository.save(statusMachine);
    }
    
    /**
     * 更新状态机
     */
    public void updateStatusMachine(StatusMachine statusMachine) {
        statusMachineRepository.update(statusMachine);
    }
    
    /**
     * 删除状态机
     */
    public void deleteStatusMachine(Long id) {
        statusMachineRepository.deleteById(id);
    }
}