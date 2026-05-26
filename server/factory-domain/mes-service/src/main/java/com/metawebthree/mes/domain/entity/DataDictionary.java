package com.metawebthree.mes.domain.entity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * 数据字典
 * 用于实现配置化MES的数据字典/下拉选项配置
 */
public class DataDictionary {
    
    private Long id;
    
    /**
     * 字典编码（唯一标识）
     */
    private String dictCode;
    
    /**
     * 字典名称
     */
    private String dictName;
    
    /**
     * 字典描述
     */
    private String description;
    
    /**
     * 状态: ACTIVE, INACTIVE
     */
    private DictStatus status;
    
    /**
     * 排序序号
     */
    private Integer sortOrder;
    
    /**
     * 创建时间
     */
    private LocalDateTime createdAt;
    
    /**
     * 更新时间
     */
    private LocalDateTime updatedAt;
    
    /**
     * 字典项列表
     */
    private List<DataDictionaryItem> items;
    
    public enum DictStatus {
        ACTIVE, INACTIVE
    }
    
    /**
     * 创建数据字典
     */
    public void create(String dictCode, String dictName, String description) {
        this.dictCode = dictCode;
        this.dictName = dictName;
        this.description = description;
        this.status = DictStatus.ACTIVE;
        this.items = new ArrayList<>();
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }
    
    /**
     * 添加字典项
     */
    public DataDictionaryItem addItem(String itemCode, String itemLabel, Integer sortOrder) {
        DataDictionaryItem item = new DataDictionaryItem();
        item.create(this.id, itemCode, itemLabel, sortOrder);
        this.items.add(item);
        return item;
    }
    
    /**
     * 移除字典项
     */
    public void removeItem(String itemCode) {
        this.items.removeIf(item -> item.getItemCode().equals(itemCode));
    }
    
    /**
     * 获取启用的字典项
     */
    public List<DataDictionaryItem> getActiveItems() {
        return this.items.stream()
                .filter(item -> item.getStatus() == DataDictionaryItem.ItemStatus.ACTIVE)
                .sorted((a, b) -> a.getSortOrder().compareTo(b.getSortOrder()))
                .toList();
    }
    
    // Getters and Setters
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
    
    /**
     * 数据字典项
     */
    public static class DataDictionaryItem {
        
        private Long id;
        
        /**
         * 所属字典ID
         */
        private Long dictId;
        
        /**
         * 选项编码
         */
        private String itemCode;
        
        /**
         * 选项标签
         */
        private String itemLabel;
        
        /**
         * 父级选项编码（用于级联选择）
         */
        private String parentItemCode;
        
        /**
         * 排序序号
         */
        private Integer sortOrder;
        
        /**
         * 状态: ACTIVE, INACTIVE
         */
        private ItemStatus status;
        
        /**
         * 创建时间
         */
        private LocalDateTime createdAt;
        
        public enum ItemStatus {
            ACTIVE, INACTIVE
        }
        
        public void create(Long dictId, String itemCode, String itemLabel, Integer sortOrder) {
            this.dictId = dictId;
            this.itemCode = itemCode;
            this.itemLabel = itemLabel;
            this.sortOrder = sortOrder;
            this.status = ItemStatus.ACTIVE;
            this.createdAt = LocalDateTime.now();
        }
        
        // Getters and Setters
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