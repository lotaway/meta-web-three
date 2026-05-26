package com.metawebthree.mes.interfaces.dto;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

public class DataDictionaryDTO {
    
    private Long id;
    private String dictCode;
    private String dictName;
    private String description;
    private String status;
    private List<DataDictionaryItemDTO> items;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    public static DataDictionaryDTO fromEntity(
            com.metawebthree.mes.domain.entity.DataDictionary entity) {
        if (entity == null) return null;
        
        DataDictionaryDTO dto = new DataDictionaryDTO();
        dto.setId(entity.getId());
        dto.setDictCode(entity.getDictCode());
        dto.setDictName(entity.getDictName());
        dto.setDescription(entity.getDescription());
        dto.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        dto.setCreatedAt(entity.getCreatedAt());
        dto.setUpdatedAt(entity.getUpdatedAt());
        
        if (entity.getItems() != null) {
            dto.setItems(entity.getItems().stream()
                    .map(DataDictionaryItemDTO::fromEntity)
                    .collect(Collectors.toList()));
        }
        
        return dto;
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getDictCode() { return dictCode; }
    public void setDictCode(String dictCode) { this.dictCode = dictCode; }
    public String getDictName() { return dictName; }
    public void setDictName(String dictName) { this.dictName = dictName; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public List<DataDictionaryItemDTO> getItems() { return items; }
    public void setItems(List<DataDictionaryItemDTO> items) { this.items = items; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    
    public static class DataDictionaryItemDTO {
        private Long id;
        private String itemCode;
        private String itemLabel;
        private String parentItemCode;
        private Integer sortOrder;
        private String status;
        
        public static DataDictionaryItemDTO fromEntity(
                com.metawebthree.mes.domain.entity.DataDictionary.DataDictionaryItem item) {
            if (item == null) return null;
            
            DataDictionaryItemDTO dto = new DataDictionaryItemDTO();
            dto.setId(item.getId());
            dto.setItemCode(item.getItemCode());
            dto.setItemLabel(item.getItemLabel());
            dto.setParentItemCode(item.getParentItemCode());
            dto.setSortOrder(item.getSortOrder());
            dto.setStatus(item.getStatus() != null ? item.getStatus().name() : null);
            return dto;
        }
        
        public Long getId() { return id; }
        public void setId(Long id) { this.id = id; }
        public String getItemCode() { return itemCode; }
        public void setItemCode(String itemCode) { this.itemCode = itemCode; }
        public String getItemLabel() { return itemLabel; }
        public void setItemLabel(String itemLabel) { this.itemLabel = itemLabel; }
        public String getParentItemCode() { return parentItemCode; }
        public void setParentItemCode(String parentItemCode) { this.parentItemCode = parentItemCode; }
        public Integer getSortOrder() { return sortOrder; }
        public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
    }
}