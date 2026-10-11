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