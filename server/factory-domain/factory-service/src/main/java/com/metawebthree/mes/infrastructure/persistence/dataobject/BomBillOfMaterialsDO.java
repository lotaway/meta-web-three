package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_bom")
public class BomBillOfMaterialsDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String bomCode;
    private String productCode;
    private String productName;
    private String version;
    private String versionStatus;
    private LocalDateTime effectiveDate;
    private LocalDateTime expiryDate;
    private String bomType;
    private String processRouteId;
    private String description;
    private String status;
    private Integer itemCount;
    private String previousVersion;
    private String changeReason;
    private String createdBy;
    private String updatedBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    // Explicit getters and setters (Lombok annotation processor not working)
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getBomCode() { return bomCode; }
    public void setBomCode(String bomCode) { this.bomCode = bomCode; }
    public String getProductCode() { return productCode; }
    public void setProductCode(String productCode) { this.productCode = productCode; }
    public String getProductName() { return productName; }
    public void setProductName(String productName) { this.productName = productName; }
    public String getVersion() { return version; }
    public void setVersion(String version) { this.version = version; }
    public String getVersionStatus() { return versionStatus; }
    public void setVersionStatus(String versionStatus) { this.versionStatus = versionStatus; }
    public LocalDateTime getEffectiveDate() { return effectiveDate; }
    public void setEffectiveDate(LocalDateTime effectiveDate) { this.effectiveDate = effectiveDate; }
    public LocalDateTime getExpiryDate() { return expiryDate; }
    public void setExpiryDate(LocalDateTime expiryDate) { this.expiryDate = expiryDate; }
    public String getBomType() { return bomType; }
    public void setBomType(String bomType) { this.bomType = bomType; }
    public String getProcessRouteId() { return processRouteId; }
    public void setProcessRouteId(String processRouteId) { this.processRouteId = processRouteId; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public Integer getItemCount() { return itemCount; }
    public void setItemCount(Integer itemCount) { this.itemCount = itemCount; }
    public String getPreviousVersion() { return previousVersion; }
    public void setPreviousVersion(String previousVersion) { this.previousVersion = previousVersion; }
    public String getChangeReason() { return changeReason; }
    public void setChangeReason(String changeReason) { this.changeReason = changeReason; }
    public String getCreatedBy() { return createdBy; }
    public void setCreatedBy(String createdBy) { this.createdBy = createdBy; }
    public String getUpdatedBy() { return updatedBy; }
    public void setUpdatedBy(String updatedBy) { this.updatedBy = updatedBy; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}