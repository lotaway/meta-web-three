package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * 设备点检记录
 */
public class EquipmentChecklistRecord {
    
    private Long id;
    private String recordCode;
    private Long equipmentId;
    private String equipmentCode;
    private Long templateId;
    private String templateCode;
    private LocalDateTime checkPlanTime;
    private LocalDateTime checkActualTime;
    private String checkerId;
    private String checkerName;
    private RecordStatus status;
    private Integer totalItems;
    private Integer checkedItems;
    private Integer abnormalItems;
    private String checkResult;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    private Map<String, Object> itemResults = new HashMap<>();
    
    public enum RecordStatus {
        PENDING, IN_PROGRESS, COMPLETED, ABNORMAL, CANCELLED
    }
    
    public static EquipmentChecklistRecord create(String recordCode, Long equipmentId, 
            String equipmentCode, Long templateId, String templateCode, LocalDateTime checkPlanTime) {
        EquipmentChecklistRecord record = new EquipmentChecklistRecord();
        record.recordCode = recordCode;
        record.equipmentId = equipmentId;
        record.equipmentCode = equipmentCode;
        record.templateId = templateId;
        record.templateCode = templateCode;
        record.checkPlanTime = checkPlanTime;
        record.status = RecordStatus.PENDING;
        record.totalItems = 0;
        record.checkedItems = 0;
        record.abnormalItems = 0;
        record.createdAt = LocalDateTime.now();
        record.updatedAt = LocalDateTime.now();
        return record;
    }
    
    public void startCheck(String checkerId, String checkerName) {
        this.checkerId = checkerId;
        this.checkerName = checkerName;
        this.checkActualTime = LocalDateTime.now();
        this.status = RecordStatus.IN_PROGRESS;
        this.updatedAt = LocalDateTime.now();
    }
    
    public void recordItemResult(String itemCode, Object value, boolean isAbnormal, String remark) {
        this.itemResults.put(itemCode + "_value", value);
        this.itemResults.put(itemCode + "_abnormal", isAbnormal);
        this.itemResults.put(itemCode + "_remark", remark);
        
        this.checkedItems++;
        if (isAbnormal) {
            this.abnormalItems++;
        }
    }
    
    public void complete() {
        this.status = this.abnormalItems > 0 ? RecordStatus.ABNORMAL : RecordStatus.COMPLETED;
        
        if (this.abnormalItems == 0) {
            this.checkResult = "OK";
        } else if (this.abnormalItems <= this.totalItems * 0.3) {
            this.checkResult = "WARNING";
        } else {
            this.checkResult = "ABNORMAL";
        }
        
        this.updatedAt = LocalDateTime.now();
    }
    
    public void cancel() {
        this.status = RecordStatus.CANCELLED;
        this.updatedAt = LocalDateTime.now();
    }
    
    public boolean isOverdue() {
        if (this.checkPlanTime == null) {
            return false;
        }
        return LocalDateTime.now().isAfter(this.checkPlanTime.plusHours(24));
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getRecordCode() { return recordCode; }
    public void setRecordCode(String recordCode) { this.recordCode = recordCode; }
    public Long getEquipmentId() { return equipmentId; }
    public void setEquipmentId(Long equipmentId) { this.equipmentId = equipmentId; }
    public String getEquipmentCode() { return equipmentCode; }
    public void setEquipmentCode(String equipmentCode) { this.equipmentCode = equipmentCode; }
    public Long getTemplateId() { return templateId; }
    public void setTemplateId(Long templateId) { this.templateId = templateId; }
    public String getTemplateCode() { return templateCode; }
    public void setTemplateCode(String templateCode) { this.templateCode = templateCode; }
    public LocalDateTime getCheckPlanTime() { return checkPlanTime; }
    public void setCheckPlanTime(LocalDateTime checkPlanTime) { this.checkPlanTime = checkPlanTime; }
    public LocalDateTime getCheckActualTime() { return checkActualTime; }
    public void setCheckActualTime(LocalDateTime checkActualTime) { this.checkActualTime = checkActualTime; }
    public String getCheckerId() { return checkerId; }
    public void setCheckerId(String checkerId) { this.checkerId = checkerId; }
    public String getCheckerName() { return checkerName; }
    public void setCheckerName(String checkerName) { this.checkerName = checkerName; }
    public RecordStatus getStatus() { return status; }
    public void setStatus(RecordStatus status) { this.status = status; }
    public Integer getTotalItems() { return totalItems; }
    public void setTotalItems(Integer totalItems) { this.totalItems = totalItems; }
    public Integer getCheckedItems() { return checkedItems; }
    public void setCheckedItems(Integer checkedItems) { this.checkedItems = checkedItems; }
    public Integer getAbnormalItems() { return abnormalItems; }
    public void setAbnormalItems(Integer abnormalItems) { this.abnormalItems = abnormalItems; }
    public String getCheckResult() { return checkResult; }
    public void setCheckResult(String checkResult) { this.checkResult = checkResult; }
    public String getRemark() { return remark; }
    public void setRemark(String remark) { this.remark = remark; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    public Map<String, Object> getItemResults() { return itemResults; }
    public void setItemResults(Map<String, Object> itemResults) { this.itemResults = itemResults; }
}