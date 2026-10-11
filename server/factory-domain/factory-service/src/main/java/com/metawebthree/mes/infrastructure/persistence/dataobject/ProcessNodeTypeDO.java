package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_process_node_type")
public class ProcessNodeTypeDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    
    private String nodeTypeCode;
    private String nodeTypeName;
    private String category;
    private String icon;
    private String configSchema;
    private String description;
    private Boolean enabled;
    private Integer sortOrder;
    private Long createdBy;
    private LocalDateTime createdAt;
    private Boolean deleted;

    // Explicit getters and setters (Lombok annotation processor not working)
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getNodeTypeCode() { return nodeTypeCode; }
    public void setNodeTypeCode(String nodeTypeCode) { this.nodeTypeCode = nodeTypeCode; }
    public String getNodeTypeName() { return nodeTypeName; }
    public void setNodeTypeName(String nodeTypeName) { this.nodeTypeName = nodeTypeName; }
    public String getCategory() { return category; }
    public void setCategory(String category) { this.category = category; }
    public String getIcon() { return icon; }
    public void setIcon(String icon) { this.icon = icon; }
    public String getConfigSchema() { return configSchema; }
    public void setConfigSchema(String configSchema) { this.configSchema = configSchema; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public Boolean getEnabled() { return enabled; }
    public void setEnabled(Boolean enabled) { this.enabled = enabled; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    public Long getCreatedBy() { return createdBy; }
    public void setCreatedBy(Long createdBy) { this.createdBy = createdBy; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public Boolean getDeleted() { return deleted; }
    public void setDeleted(Boolean deleted) { this.deleted = deleted; }
}