package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

public class Workstation {
    
    public enum WorkstationType {
        ASSEMBLY,        // 装配工位
        INSPECTION,      // 质检工位
        PACKAGING,       // 包装工位
        STORAGE,         // 存储工位
        MATERIAL_PREP,   // 备料工位
        TESTING,         // 测试工位
        REWORK,          // 返工工位
        OTHER            // 其他
    }
    
    public enum WorkstationStatus {
        ACTIVE,
        INACTIVE,
        MAINTENANCE,
        FAULT
    }
    
    private Long id;
    private String workstationCode;
    private String workstationName;
    private String workshopId;
    private String workshopName;
    private WorkstationType type;
    private WorkstationStatus status;
    private String location;
    private Integer capacity;
    private String description;
    
    // 工位关联
    private List<Long> equipmentIds;
    private List<String> equipmentCodes;
    private List<Long> toolIds;
    private List<String> toolNames;
    private List<String> operatorIds;
    private List<String> operatorNames;
    
    // 扩展字段
    private Map<String, Object> extensionFields;
    
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public void create(String workstationCode, String workstationName, 
                      String workshopId, WorkstationType type) {
        this.workstationCode = workstationCode;
        this.workstationName = workstationName;
        this.workshopId = workshopId;
        this.type = type;
        this.status = WorkstationStatus.ACTIVE;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    public void activate() {
        if (this.status == WorkstationStatus.INACTIVE) {
            this.status = WorkstationStatus.ACTIVE;
            this.updatedAt = LocalDateTime.now();
        }
    }
    
    public void deactivate() {
        if (this.status == WorkstationStatus.ACTIVE) {
            this.status = WorkstationStatus.INACTIVE;
            this.updatedAt = LocalDateTime.now();
        }
    }
    
    public void setMaintenance() {
        if (this.status == WorkstationStatus.ACTIVE || this.status == WorkstationStatus.FAULT) {
            this.status = WorkstationStatus.MAINTENANCE;
            this.updatedAt = LocalDateTime.now();
        }
    }
    
    public void setFault() {
        if (this.status != WorkstationStatus.INACTIVE) {
            this.status = WorkstationStatus.FAULT;
            this.updatedAt = LocalDateTime.now();
        }
    }
    
    public void bindEquipment(Long equipmentId, String equipmentCode) {
        if (this.equipmentIds == null) {
            this.equipmentIds = new java.util.ArrayList<>();
            this.equipmentCodes = new java.util.ArrayList<>();
        }
        if (!this.equipmentIds.contains(equipmentId)) {
            this.equipmentIds.add(equipmentId);
            this.equipmentCodes.add(equipmentCode);
            this.updatedAt = LocalDateTime.now();
        }
    }
    
    public void unbindEquipment(Long equipmentId) {
        if (this.equipmentIds != null) {
            int index = this.equipmentIds.indexOf(equipmentId);
            if (index >= 0) {
                this.equipmentIds.remove(index);
                if (this.equipmentCodes != null && index < this.equipmentCodes.size()) {
                    this.equipmentCodes.remove(index);
                }
                this.updatedAt = LocalDateTime.now();
            }
        }
    }
    
    public void bindTool(Long toolId, String toolName) {
        if (this.toolIds == null) {
            this.toolIds = new java.util.ArrayList<>();
            this.toolNames = new java.util.ArrayList<>();
        }
        if (!this.toolIds.contains(toolId)) {
            this.toolIds.add(toolId);
            this.toolNames.add(toolName);
            this.updatedAt = LocalDateTime.now();
        }
    }
    
    public void unbindTool(Long toolId) {
        if (this.toolIds != null) {
            int index = this.toolIds.indexOf(toolId);
            if (index >= 0) {
                this.toolIds.remove(index);
                if (this.toolNames != null && index < this.toolNames.size()) {
                    this.toolNames.remove(index);
                }
                this.updatedAt = LocalDateTime.now();
            }
        }
    }
    
    public void bindOperator(String operatorId, String operatorName) {
        if (this.operatorIds == null) {
            this.operatorIds = new java.util.ArrayList<>();
            this.operatorNames = new java.util.ArrayList<>();
        }
        if (!this.operatorIds.contains(operatorId)) {
            this.operatorIds.add(operatorId);
            this.operatorNames.add(operatorName);
            this.updatedAt = LocalDateTime.now();
        }
    }
    
    public void unbindOperator(String operatorId) {
        if (this.operatorIds != null) {
            int index = this.operatorIds.indexOf(operatorId);
            if (index >= 0) {
                this.operatorIds.remove(index);
                if (this.operatorNames != null && index < this.operatorNames.size()) {
                    this.operatorNames.remove(index);
                }
                this.updatedAt = LocalDateTime.now();
            }
        }
    }
    
    // Getters and Setters
    public Long getId() {
        return id;
    }
    
    public void setId(Long id) {
        this.id = id;
    }
    
    public String getWorkstationCode() {
        return workstationCode;
    }
    
    public void setWorkstationCode(String workstationCode) {
        this.workstationCode = workstationCode;
    }
    
    public String getWorkstationName() {
        return workstationName;
    }
    
    public void setWorkstationName(String workstationName) {
        this.workstationName = workstationName;
    }
    
    public String getWorkshopId() {
        return workshopId;
    }
    
    public void setWorkshopId(String workshopId) {
        this.workshopId = workshopId;
    }
    
    public String getWorkshopName() {
        return workshopName;
    }
    
    public void setWorkshopName(String workshopName) {
        this.workshopName = workshopName;
    }
    
    public WorkstationType getType() {
        return type;
    }
    
    public void setType(WorkstationType type) {
        this.type = type;
    }
    
    public WorkstationStatus getStatus() {
        return status;
    }
    
    public void setStatus(WorkstationStatus status) {
        this.status = status;
    }
    
    public String getLocation() {
        return location;
    }
    
    public void setLocation(String location) {
        this.location = location;
    }
    
    public Integer getCapacity() {
        return capacity;
    }
    
    public void setCapacity(Integer capacity) {
        this.capacity = capacity;
    }
    
    public String getDescription() {
        return description;
    }
    
    public void setDescription(String description) {
        this.description = description;
    }
    
    public List<Long> getEquipmentIds() {
        return equipmentIds;
    }
    
    public void setEquipmentIds(List<Long> equipmentIds) {
        this.equipmentIds = equipmentIds;
    }
    
    public List<String> getEquipmentCodes() {
        return equipmentCodes;
    }
    
    public void setEquipmentCodes(List<String> equipmentCodes) {
        this.equipmentCodes = equipmentCodes;
    }
    
    public List<Long> getToolIds() {
        return toolIds;
    }
    
    public void setToolIds(List<Long> toolIds) {
        this.toolIds = toolIds;
    }
    
    public List<String> getToolNames() {
        return toolNames;
    }
    
    public void setToolNames(List<String> toolNames) {
        this.toolNames = toolNames;
    }
    
    public List<String> getOperatorIds() {
        return operatorIds;
    }
    
    public void setOperatorIds(List<String> operatorIds) {
        this.operatorIds = operatorIds;
    }
    
    public List<String> getOperatorNames() {
        return operatorNames;
    }
    
    public void setOperatorNames(List<String> operatorNames) {
        this.operatorNames = operatorNames;
    }
    
    public Map<String, Object> getExtensionFields() {
        return extensionFields;
    }
    
    public void setExtensionFields(Map<String, Object> extensionFields) {
        this.extensionFields = extensionFields;
    }
    
    public LocalDateTime getCreatedAt() {
        return createdAt;
    }
    
    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }
    
    public LocalDateTime getUpdatedAt() {
        return updatedAt;
    }
    
    public void setUpdatedAt(LocalDateTime updatedAt) {
        this.updatedAt = updatedAt;
    }
}