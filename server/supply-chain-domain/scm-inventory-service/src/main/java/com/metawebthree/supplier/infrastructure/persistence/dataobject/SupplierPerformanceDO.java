package com.metawebthree.supplier.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

/**
 * 供应商绩效评估数据对象
 */
@Data
@TableName("supplier_performance")
public class SupplierPerformanceDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private Long supplierId;
    
    private String supplierCode;
    
    private String supplierName;
    
    /**
     * 评估周期（开始日期）
     */
    private LocalDate periodStart;
    
    /**
     * 评估周期（结束日期）
     */
    private LocalDate periodEnd;
    
    /**
     * 交货及时率 (0-100)
     */
    private BigDecimal onTimeDeliveryRate;
    
    /**
     * 质量合格率 (0-100)
     */
    private BigDecimal qualityPassRate;
    
    /**
     * 价格竞争力评分 (0-100)
     */
    private BigDecimal priceCompetitivenessScore;
    
    /**
     * 综合评分 (0-100)
     */
    private BigDecimal overallScore;
    
    /**
     * 评估等级 (A/B/C/D)
     */
    private String assessmentLevel;
    
    /**
     * 订单总数
     */
    private Integer totalOrders;
    
    /**
     * 准时交付订单数
     */
    private Integer onTimeDeliveryCount;
    
    /**
     * 质检合格数量
     */
    private Integer qualifiedCount;
    
    /**
     * 质检总数量
     */
    private Integer totalQualityCheckCount;
    
    /**
     * 市场平均价格
     */
    private BigDecimal marketAvgPrice;
    
    /**
     * 供应商价格
     */
    private BigDecimal supplierPrice;
    
    /**
     * 备注
     */
    private String remark;
    
    /**
     * 评估人
     */
    private String assessor;
    
    /**
     * 评估时间
     */
    private LocalDateTime assessmentDate;
    
    private LocalDateTime createdAt;
    
    private LocalDateTime updatedAt;
}