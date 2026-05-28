package com.metawebthree.mes.domain.exception;

public class EquipmentException extends RuntimeException {
    
    private final String code;
    private final String details;
    
    public EquipmentException(String code, String message) {
        super(message);
        this.code = code;
        this.details = null;
    }
    
    public EquipmentException(String code, String message, String details) {
        super(message);
        this.code = code;
        this.details = details;
    }
    
    public EquipmentException(String code, String message, Throwable cause) {
        super(message, cause);
        this.code = code;
        this.details = null;
    }
    
    public String getCode() {
        return code;
    }
    
    public String getDetails() {
        return details;
    }
    
    public static EquipmentException notFound(Long id) {
        return new EquipmentException("EQUIPMENT_NOT_FOUND", 
            "设备不存在", "ID: " + id);
    }
    
    public static EquipmentException notFound(String equipmentCode) {
        return new EquipmentException("EQUIPMENT_NOT_FOUND", 
            "设备不存在", "编码: " + equipmentCode);
    }
    
    public static EquipmentException invalidState(String currentState, String operation) {
        return new EquipmentException("EQUIPMENT_INVALID_STATE", 
            "设备状态不允许执行该操作", 
            "当前状态: " + currentState + ", 操作: " + operation);
    }
    
    public static EquipmentException stateTransitionFailed(String fromState, String toState) {
        return new EquipmentException("EQUIPMENT_STATE_TRANSITION_FAILED", 
            "设备状态转换失败", 
            "从 " + fromState + " 到 " + toState + " 的转换无效");
    }
}