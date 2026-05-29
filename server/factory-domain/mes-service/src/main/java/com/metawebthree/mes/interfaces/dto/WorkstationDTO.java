package com.metawebthree.mes.interfaces.dto;

import java.time.LocalDateTime;
import java.util.List;

import com.metawebthree.mes.domain.entity.Workstation;

public class WorkstationDTO {
    
    private Long id;
    private String workstationCode;
    private String workstationName;
    private String workshopId;
    private String workshopName;
    private String type;
    private String status;
    private String location;
    private Integer capacity;
    private String description;
    private List<Long> equipmentIds;
    private List<String> equipmentCodes;
    private List<Long> toolIds;
    private List<String> toolNames;
    private List<String> operatorIds;
    private List<String> operatorNames;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public static WorkstationDTO fromEntity(Workstation entity) {
        if (entity == null) return null;
        
        WorkstationDTO dto = new WorkstationDTO();
        dto.setId(entity.getId());
        dto.setWorkstationCode(entity.getWorkstationCode());
        dto.setWorkstationName(entity.getWorkstationName());
        dto.setWorkshopId(entity.getWorkshopId());
        dto.setWorkshopName(entity.getWorkshopName());
        dto.setType(entity.getType() != null ? entity.getType().name() : null);
        dto.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        dto.setLocation(entity.getLocation());
        dto.setCapacity(entity.getCapacity());
        dto.setDescription(entity.getDescription());
        dto.setEquipmentIds(entity.getEquipmentIds());
        dto.setEquipmentCodes(entity.getEquipmentCodes());
        dto.setToolIds(entity.getToolIds());
        dto.setToolNames(entity.getToolNames());
        dto.setOperatorIds(entity.getOperatorIds());
        dto.setOperatorNames(entity.getOperatorNames());
        dto.setCreatedAt(entity.getCreatedAt());
        dto.setUpdatedAt(entity.getUpdatedAt());
        
        return dto;
    }
    
    public Workstation toEntity() {
        Workstation workstation = new Workstation();
        if (this.id != null) {
            workstation.setId(this.id);
        }
        workstation.setWorkstationCode(this.workstationCode);
        workstation.setWorkstationName(this.workstationName);
        workstation.setWorkshopId(this.workshopId);
        workstation.setWorkshopName(this.workshopName);
        workstation.setLocation(this.location);
        workstation.setCapacity(this.capacity);
        workstation.setDescription(this.description);
        workstation.setEquipmentIds(this.equipmentIds);
        workstation.setEquipmentCodes(this.equipmentCodes);
        workstation.setToolIds(this.toolIds);
        workstation.setToolNames(this.toolNames);
        workstation.setOperatorIds(this.operatorIds);
        workstation.setOperatorNames(this.operatorNames);
        
        return workstation;
    }
    
    // ========== Request DTOs ==========
    
    public static class CreateRequest {
        private String workstationCode;
        private String workstationName;
        private String workshopId;
        private String workshopName;
        private String type;
        private String location;
        private Integer capacity;
        private String description;
        
        public String getWorkstationCode() { return workstationCode; }
        public void setWorkstationCode(String workstationCode) { this.workstationCode = workstationCode; }
        public String getWorkstationName() { return workstationName; }
        public void setWorkstationName(String workstationName) { this.workstationName = workstationName; }
        public String getWorkshopId() { return workshopId; }
        public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
        public String getWorkshopName() { return workshopName; }
        public void setWorkshopName(String workshopName) { this.workshopName = workshopName; }
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        public String getLocation() { return location; }
        public void setLocation(String location) { this.location = location; }
        public Integer getCapacity() { return capacity; }
        public void setCapacity(Integer capacity) { this.capacity = capacity; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
    }
    
    public static class UpdateRequest {
        private String workstationName;
        private String workshopId;
        private String workshopName;
        private String type;
        private String status;
        private String location;
        private Integer capacity;
        private String description;
        
        public String getWorkstationName() { return workstationName; }
        public void setWorkstationName(String workstationName) { this.workstationName = workstationName; }
        public String getWorkshopId() { return workshopId; }
        public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
        public String getWorkshopName() { return workshopName; }
        public void setWorkshopName(String workshopName) { this.workshopName = workshopName; }
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        public String getLocation() { return location; }
        public void setLocation(String location) { this.location = location; }
        public Integer getCapacity() { return capacity; }
        public void setCapacity(Integer capacity) { this.capacity = capacity; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
    }
    
    public static class BindEquipmentRequest {
        private Long equipmentId;
        private String equipmentCode;
        
        public Long getEquipmentId() { return equipmentId; }
        public void setEquipmentId(Long equipmentId) { this.equipmentId = equipmentId; }
        public String getEquipmentCode() { return equipmentCode; }
        public void setEquipmentCode(String equipmentCode) { this.equipmentCode = equipmentCode; }
    }
    
    public static class BindToolRequest {
        private Long toolId;
        private String toolName;
        
        public Long getToolId() { return toolId; }
        public void setToolId(Long toolId) { this.toolId = toolId; }
        public String getToolName() { return toolName; }
        public void setToolName(String toolName) { this.toolName = toolName; }
    }
    
    public static class BindOperatorRequest {
        private String operatorId;
        private String operatorName;
        
        public String getOperatorId() { return operatorId; }
        public void setOperatorId(String operatorId) { this.operatorId = operatorId; }
        public String getOperatorName() { return operatorName; }
        public void setOperatorName(String operatorName) { this.operatorName = operatorName; }
    }
    
    // ========== Getters and Setters ==========
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getWorkstationCode() { return workstationCode; }
    public void setWorkstationCode(String workstationCode) { this.workstationCode = workstationCode; }
    public String getWorkstationName() { return workstationName; }
    public void setWorkstationName(String workstationName) { this.workstationName = workstationName; }
    public String getWorkshopId() { return workshopId; }
    public void setWorkshopId(String workshopId) { this.workshopId = workshopId; }
    public String getWorkshopName() { return workshopName; }
    public void setWorkshopName(String workshopName) { this.workshopName = workshopName; }
    public String getType() { return type; }
    public void setType(String type) { this.type = type; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getLocation() { return location; }
    public void setLocation(String location) { this.location = location; }
    public Integer getCapacity() { return capacity; }
    public void setCapacity(Integer capacity) { this.capacity = capacity; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public List<Long> getEquipmentIds() { return equipmentIds; }
    public void setEquipmentIds(List<Long> equipmentIds) { this.equipmentIds = equipmentIds; }
    public List<String> getEquipmentCodes() { return equipmentCodes; }
    public void setEquipmentCodes(List<String> equipmentCodes) { this.equipmentCodes = equipmentCodes; }
    public List<Long> getToolIds() { return toolIds; }
    public void setToolIds(List<Long> toolIds) { this.toolIds = toolIds; }
    public List<String> getToolNames() { return toolNames; }
    public void setToolNames(List<String> toolNames) { this.toolNames = toolNames; }
    public List<String> getOperatorIds() { return operatorIds; }
    public void setOperatorIds(List<String> operatorIds) { this.operatorIds = operatorIds; }
    public List<String> getOperatorNames() { return operatorNames; }
    public void setOperatorNames(List<String> operatorNames) { this.operatorNames = operatorNames; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}