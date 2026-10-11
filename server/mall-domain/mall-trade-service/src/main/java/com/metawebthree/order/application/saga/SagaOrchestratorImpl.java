package com.metawebthree.order.application.saga;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.order.domain.model.SagaInstance;
import com.metawebthree.order.domain.model.SagaStep;
import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.order.infrastructure.persistence.mapper.SagaInstanceMapper;
import com.metawebthree.order.infrastructure.persistence.mapper.SagaStepMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;
import java.util.function.Function;

/**
 * Saga orchestrator implementation.
 */
@Slf4j
@Component
public class SagaOrchestratorImpl implements SagaOrchestrator {

    private final SagaInstanceMapper sagaInstanceMapper;
    private final SagaStepMapper sagaStepMapper;
    private final ObjectMapper objectMapper;

    public SagaOrchestratorImpl(SagaInstanceMapper sagaInstanceMapper,
                                 SagaStepMapper sagaStepMapper,
                                 ObjectMapper objectMapper) {
        this.sagaInstanceMapper = sagaInstanceMapper;
        this.sagaStepMapper = sagaStepMapper;
        this.objectMapper = objectMapper;
    }

    @Override
    @Transactional
    public SagaInstance startSaga(String sagaType, String bizId, List<SagaStepDefinition> steps) {
        String sagaId = UUID.randomUUID().toString();
        
        // Create saga instance
        SagaInstance instance = SagaInstance.builder()
                .sagaId(sagaId)
                .bizId(bizId)
                .sagaType(sagaType)
                .status(SagaInstance.Status.RUNNING)
                .currentStep(null)
                .startTime(LocalDateTime.now())
                .build();
        sagaInstanceMapper.insert(instance);
        
        // Create saga steps
        for (SagaStepDefinition stepDef : steps) {
            SagaStep step = SagaStep.builder()
                    .sagaId(sagaId)
                    .stepName(stepDef.getStepName())
                    .stepOrder(stepDef.getStepOrder())
                    .serviceName(stepDef.getServiceName())
                    .compensable(stepDef.isCompensable())
                    .status(SagaStep.Status.PENDING)
                    .retryCount(0)
                    .maxRetries(3)
                    .build();
            sagaStepMapper.insert(step);
        }
        
        log.info("Saga started: sagaId={}, type={}, bizId={}", sagaId, sagaType, bizId);
        return instance;
    }

    @Override
    @Transactional
    public SagaStep executeStep(String sagaId, String stepName, Object request,
                                  Function<Object, SagaStepResult> action,
                                  Function<Object, Object> compensationAction) {
        // Get saga step
        List<SagaStep> steps = sagaStepMapper.selectList(
            new LambdaQueryWrapper<SagaStep>()
                .eq(SagaStep::getSagaId, sagaId)
                .eq(SagaStep::getStepName, stepName)
        );
        
        if (steps.isEmpty()) {
            throw new IllegalArgumentException("Step not found: " + stepName);
        }
        
        SagaStep step = steps.get(0);
        step.setStatus(SagaStep.Status.RUNNING);
        step.setStartTime(LocalDateTime.now());
        
        try {
            // Serialize request data
            step.setRequestData(toJson(request));
            
            // Execute the action
            SagaStepResult result = action.apply(request);
            
            if (result.isSuccess()) {
                // Update step as completed
                step.setStatus(SagaStep.Status.COMPLETED);
                step.setResponseData(toJson(result.getData()));
                step.setCompensationData(toJson(result.getCompensationData()));
                step.setEndTime(LocalDateTime.now());
                
                // Update saga current step
                SagaInstance instance = getSagaInstance(sagaId);
                if (instance != null) {
                    instance.setCurrentStep(stepName);
                    sagaInstanceMapper.updateById(instance);
                }
                
                log.info("Saga step completed: sagaId={}, step={}", sagaId, stepName);
            } else {
                // Mark step as failed
                step.setStatus(SagaStep.Status.FAILED);
                step.setErrorMessage(result.getMessage());
                step.setEndTime(LocalDateTime.now());
                
                // Mark saga as failed
                updateSagaStatus(sagaId, SagaInstance.Status.FAILED, result.getMessage());
                
                log.error("Saga step failed: sagaId={}, step={}, error={}", sagaId, stepName, result.getMessage());
            }
            
            sagaStepMapper.updateById(step);
            return step;
            
        } catch (Exception e) {
            log.error("Error executing saga step: sagaId={}, step={}", sagaId, stepName, e);
            step.setStatus(SagaStep.Status.FAILED);
            step.setErrorMessage(e.getMessage());
            step.setEndTime(LocalDateTime.now());
            sagaStepMapper.updateById(step);
            
            updateSagaStatus(sagaId, SagaInstance.Status.FAILED, e.getMessage());
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Saga step execution failed: " + stepName, e);
        }
    }

    @Override
    @Transactional
    public boolean compensate(String sagaId) {
        log.info("Starting saga compensation: sagaId={}", sagaId);
        
        SagaInstance instance = getSagaInstance(sagaId);
        if (instance == null) {
            log.error("Saga instance not found: sagaId={}", sagaId);
            return false;
        }
        
        // Get completed steps in reverse order
        List<SagaStep> steps = sagaStepMapper.selectList(
            new LambdaQueryWrapper<SagaStep>()
                .eq(SagaStep::getSagaId, sagaId)
                .eq(SagaStep::getStatus, SagaStep.Status.COMPLETED)
                .orderByDesc(SagaStep::getStepOrder)
        );
        
        boolean allCompensated = true;
        for (SagaStep step : steps) {
            if (!step.getCompensable()) {
                log.info("Step is not compensable, skipping: step={}", step.getStepName());
                continue;
            }
            
            try {
                // Execute compensation (this should be implemented by specific saga)
                log.info("Compensating step: sagaId={}, step={}", sagaId, step.getStepName());
                
                step.setStatus(SagaStep.Status.COMPENSATED);
                step.setEndTime(LocalDateTime.now());
                sagaStepMapper.updateById(step);
                
            } catch (Exception e) {
                log.error("Compensation failed for step: sagaId={}, step={}, error={}", 
                          sagaId, step.getStepName(), e.getMessage());
                allCompensated = false;
            }
        }
        
        // Update saga status
        if (allCompensated) {
            updateSagaStatus(sagaId, SagaInstance.Status.COMPENSATED, null);
        } else {
            updateSagaStatus(sagaId, SagaInstance.Status.FAILED, "Compensation partially failed");
        }
        
        return allCompensated;
    }

    @Override
    public SagaInstance getSagaInstance(String sagaId) {
        return sagaInstanceMapper.selectOne(
            new LambdaQueryWrapper<SagaInstance>()
                .eq(SagaInstance::getSagaId, sagaId)
        );
    }

    @Override
    @Transactional
    public void updateSagaStatus(String sagaId, String status, String errorMessage) {
        SagaInstance instance = getSagaInstance(sagaId);
        if (instance != null) {
            instance.setStatus(status);
            instance.setErrorMessage(errorMessage);
            if (SagaInstance.Status.COMPLETED.equals(status) || 
                SagaInstance.Status.COMPENSATED.equals(status) ||
                SagaInstance.Status.FAILED.equals(status)) {
                instance.setEndTime(LocalDateTime.now());
            }
            sagaInstanceMapper.updateById(instance);
            
            log.info("Saga status updated: sagaId={}, status={}", sagaId, status);
        }
    }
    
    private String toJson(Object obj) {
        if (obj == null) {
            return null;
        }
        try {
            return objectMapper.writeValueAsString(obj);
        } catch (JsonProcessingException e) {
            log.error("Failed to serialize object", e);
            return null;
        }
    }
    
    private <T> T fromJson(String json, Class<T> clazz) {
        if (json == null) {
            return null;
        }
        try {
            return objectMapper.readValue(json, clazz);
        } catch (JsonProcessingException e) {
            log.error("Failed to deserialize object", e);
            return null;
        }
    }
}
