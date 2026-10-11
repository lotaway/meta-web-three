package com.metawebthree.mes.application.query;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ProcessFlowInstanceDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ProcessFlowTemplateDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.ProcessFlowInstanceMapper;
import com.metawebthree.mes.infrastructure.persistence.mapper.ProcessFlowTemplateMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
public class ProcessFlowQueryService {
    
    private final ProcessFlowInstanceMapper instanceMapper;
    private final ProcessFlowTemplateMapper templateMapper;
    
    /**
     * 启动流程实例
     */
    @Transactional
    public ProcessFlowInstanceDO startInstance(Long templateId, String businessType, 
                                                String businessKey, Long userId) {
        ProcessFlowTemplateDO template = templateMapper.selectById(templateId);
        if (template == null || !"PUBLISHED".equals(template.getStatus())) {
            throw new IllegalArgumentException("流程模板不存在或未发布");
        }
        
        ProcessFlowInstanceDO instance = new ProcessFlowInstanceDO();
        instance.setInstanceCode(generateInstanceCode());
        instance.setTemplateId(templateId);
        instance.setTemplateName(template.getTemplateName());
        instance.setBusinessType(businessType);
        instance.setBusinessKey(businessKey);
        instance.setStatus("RUNNING");
        instance.setFlowData(template.getFlowData());
        instance.setStartedAt(LocalDateTime.now());
        instance.setStartedBy(userId);
        instance.setDeleted(false);
        instance.setCreatedAt(LocalDateTime.now());
        
        instanceMapper.insert(instance);
        return instance;
    }
    
    @Transactional
    public void completeInstance(Long instanceId, Long userId) {
        ProcessFlowInstanceDO instance = instanceMapper.selectById(instanceId);
        if (instance != null) {
            instance.setStatus("COMPLETED");
            instance.setCompletedAt(LocalDateTime.now());
            instance.setCompletedBy(userId);
            instance.setUpdatedAt(LocalDateTime.now());
            instanceMapper.updateById(instance);
        }
    }
    
    @Transactional
    public void terminateInstance(Long instanceId, Long userId) {
        ProcessFlowInstanceDO instance = instanceMapper.selectById(instanceId);
        if (instance != null) {
            instance.setStatus("TERMINATED");
            instance.setCompletedAt(LocalDateTime.now());
            instance.setCompletedBy(userId);
            instance.setUpdatedAt(LocalDateTime.now());
            instanceMapper.updateById(instance);
        }
    }
    
    public ProcessFlowInstanceDO getInstance(Long instanceId) {
        return instanceMapper.selectById(instanceId);
    }
    
    public List<ProcessFlowInstanceDO> listInstances(String businessType, String status) {
        LambdaQueryWrapper<ProcessFlowInstanceDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProcessFlowInstanceDO::getDeleted, false);
        if (businessType != null && !businessType.isEmpty()) {
            wrapper.eq(ProcessFlowInstanceDO::getBusinessType, businessType);
        }
        if (status != null && !status.isEmpty()) {
            wrapper.eq(ProcessFlowInstanceDO::getStatus, status);
        }
        wrapper.orderByDesc(ProcessFlowInstanceDO::getCreatedAt);
        return instanceMapper.selectList(wrapper);
    }
    
    public List<ProcessFlowInstanceDO> listByBusinessKey(String businessType, String businessKey) {
        LambdaQueryWrapper<ProcessFlowInstanceDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ProcessFlowInstanceDO::getDeleted, false)
               .eq(ProcessFlowInstanceDO::getBusinessType, businessType)
               .eq(ProcessFlowInstanceDO::getBusinessKey, businessKey)
               .orderByDesc(ProcessFlowInstanceDO::getCreatedAt);
        return instanceMapper.selectList(wrapper);
    }
    
    private String generateInstanceCode() {
        return "PI-" + System.currentTimeMillis();
    }
}