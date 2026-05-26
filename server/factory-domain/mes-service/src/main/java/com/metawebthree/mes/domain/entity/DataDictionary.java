package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

public class DataDictionary {
    
    private Long id;
    private String dictCode;
    private String dictName;
    private String description;
    private DictStatus status;
    private Integer sortOrder;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private List<DataDictionaryItem> items;
    
    public enum DictStatus {
        ACTIVE, INACTIVE
    }
    
    public static DataDictionary create(String dictCode, String dictName, String description) {
        DataDictionary dict = new DataDictionary();
        dict.dictCode = dictCode;
        dict.dictName = dictName;
        dict.description = description;
        dict.status = DictStatus.ACTIVE;
        dict.items = new ArrayList<>();
        dict.createdAt = LocalDateTime.now();
        dict.updatedAt = LocalDateTime.now();
        return dict;
    }
    
    public DataDictionaryItem addItem(String itemCode, String itemLabel, Integer sortOrder) {
        DataDictionaryItem item = DataDictionaryItem.create(this.id, itemCode, itemLabel, sortOrder);
        this.items.add(item);
        return item;
    }
    
    public void removeItem(String itemCode) {
        this.items.removeIf(item -> item.getItemCode().equals(itemCode));
    }
    
    public List<DataDictionaryItem> getActiveItems() {
        return this.items.stream()
                .filter(item -> item.getStatus() == DataDictionaryItem.ItemStatus.ACTIVE)
                .sorted((a, b) -> a.getSortOrder().compareTo(b.getSortOrder()))
                .toList();
    }
    
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getDictCode() { return dictCode; }
    public void setDictCode(String dictCode) { this.dictCode = dictCode; }
    public String getDictName() { return dictName; }
    public void setDictName(String dictName) { this.dictName = dictName; }
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    public DictStatus getStatus() { return status; }
    public void setStatus(DictStatus status) { this.status = status; }
    public Integer getSortOrder() { return sortOrder; }
    public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
    public List<DataDictionaryItem> getItems() { return items; }
    public void setItems(List<DataDictionaryItem> items) { this.items = items; }
    
    public static class DataDictionaryItem {
        
        private Long id;
        private Long dictId;
        private String itemCode;
        private String itemLabel;
        private String parentItemCode;
        private Integer sortOrder;
        private ItemStatus status;
        private LocalDateTime createdAt;
        
        public enum ItemStatus {
            ACTIVE, INACTIVE
        }
        
        public static DataDictionaryItem create(Long dictId, String itemCode, String itemLabel, Integer sortOrder) {
            DataDictionaryItem item = new DataDictionaryItem();
            item.dictId = dictId;
            item.itemCode = itemCode;
            item.itemLabel = itemLabel;
            item.sortOrder = sortOrder;
            item.status = ItemStatus.ACTIVE;
            item.createdAt = LocalDateTime.now();
            return item;
        }
        
        public Long getId() { return id; }
        public void setId(Long id) { this.id = id; }
        public Long getDictId() { return dictId; }
        public void setDictId(Long dictId) { this.dictId = dictId; }
        public String getItemCode() { return itemCode; }
        public void setItemCode(String itemCode) { this.itemCode = itemCode; }
        public String getItemLabel() { return itemLabel; }
        public void setItemLabel(String itemLabel) { this.itemLabel = itemLabel; }
        public String getParentItemCode() { return parentItemCode; }
        public void setParentItemCode(String parentItemCode) { this.parentItemCode = parentItemCode; }
        public Integer getSortOrder() { return sortOrder; }
        public void setSortOrder(Integer sortOrder) { this.sortOrder = sortOrder; }
        public ItemStatus getStatus() { return status; }
        public void setStatus(ItemStatus status) { this.status = status; }
        public LocalDateTime getCreatedAt() { return createdAt; }
    }
}