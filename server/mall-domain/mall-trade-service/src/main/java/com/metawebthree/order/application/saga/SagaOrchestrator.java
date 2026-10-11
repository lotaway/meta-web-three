package com.metawebthree.order.application.saga;

import com.metawebthree.order.domain.model.SagaInstance;
import com.metawebthree.order.domain.model.SagaStep;

import java.util.List;
import java.util.function.Function;

/**
 * Saga orchestrator interface.
 */
public interface SagaOrchestrator {
    
    /**
     * Start a new saga transaction.
     */
    SagaInstance startSaga(String sagaType, String bizId, List<SagaStepDefinition> steps);
    
    /**
     * Execute a specific step.
     */
    SagaStep executeStep(String sagaId, String stepName, Object request, 
                         Function<Object, SagaStepResult> action,
                         Function<Object, Object> compensationAction);
    
    /**
     * Compensate all completed steps in reverse order.
     */
    boolean compensate(String sagaId);
    
    /**
     * Get saga instance by saga ID.
     */
    SagaInstance getSagaInstance(String sagaId);
    
    /**
     * Update saga instance status.
     */
    void updateSagaStatus(String sagaId, String status, String errorMessage);
    
    /**
     * Saga step definition.
     */
    class SagaStepDefinition {
        private String stepName;
        private String serviceName;
        private int stepOrder;
        private boolean compensable;
        
        public SagaStepDefinition(String stepName, String serviceName, int stepOrder, boolean compensable) {
            this.stepName = stepName;
            this.serviceName = serviceName;
            this.stepOrder = stepOrder;
            this.compensable = compensable;
        }
        
        public String getStepName() { return stepName; }
        public String getServiceName() { return serviceName; }
        public int getStepOrder() { return stepOrder; }
        public boolean isCompensable() { return compensable; }
    }
    
    /**
     * Saga step result.
     */
    class SagaStepResult {
        private boolean success;
        private String message;
        private Object data;
        private Object compensationData;
        
        public static SagaStepResult success(Object data, Object compensationData) {
            SagaStepResult result = new SagaStepResult();
            result.success = true;
            result.data = data;
            result.compensationData = compensationData;
            return result;
        }
        
        public static SagaStepResult failure(String message) {
            SagaStepResult result = new SagaStepResult();
            result.success = false;
            result.message = message;
            return result;
        }
        
        public boolean isSuccess() { return success; }
        public String getMessage() { return message; }
        public Object getData() { return data; }
        public Object getCompensationData() { return compensationData; }
    }
}
