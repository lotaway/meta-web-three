package com.metawebthree.dataanalysis.domain.entity;

import java.time.LocalDateTime;

/**
 * Entity for product sales statistics (hot products)
 */
public class ProductSalesDO {
    private Long id;
    private String date;
    private Long productId;
    private String productName;
    private String category;
    private Long salesCount;
    private Long salesAmount;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getDate() { return date; }
    public void setDate(String date) { this.date = date; }
    public Long getProductId() { return productId; }
    public void setProductId(Long productId) { this.productId = productId; }
    public String getProductName() { return productName; }
    public void setProductName(String productName) { this.productName = productName; }
    public String getCategory() { return category; }
    public void setCategory(String category) { this.category = category; }
    public Long getSalesCount() { return salesCount; }
    public void setSalesCount(Long salesCount) { this.salesCount = salesCount; }
    public Long getSalesAmount() { return salesAmount; }
    public void setSalesAmount(Long salesAmount) { this.salesAmount = salesAmount; }
    public LocalDateTime getCreateTime() { return createTime; }
    public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
    public LocalDateTime getUpdateTime() { return updateTime; }
    public void setUpdateTime(LocalDateTime updateTime) { this.updateTime = updateTime; }
}