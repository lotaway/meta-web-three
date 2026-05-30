package com.metawebthree.supplier.application.dto;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;

/**
 * 供应商绩效评估 DTO
 */
@Data
public class SupplierPerformanceDTO {
    private Long id;
    private Long supplierId;
    private String supplierCode;
    private String supplierName;
    private LocalDate periodStart;
    private LocalDate periodEnd;
    private BigDecimal onTimeDeliveryRate;
    private BigDecimal qualityPassRate;
    private BigDecimal priceCompetitivenessScore;
    private BigDecimal overallScore;
    private String assessmentLevel;
    private Integer totalOrders;
    private Integer onTimeDeliveryCount;
    private Integer qualifiedCount;
    private Integer totalQualityCheckCount;
    private BigDecimal marketAvgPrice;
    private BigDecimal supplierPrice;
    private String remark;
    private String assessor;
    private LocalDateTime assessmentDate;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}

/**
 * 供应商绩效评估请求 DTO
 */
@Data
class SupplierPerformanceRequest {
    private Long supplierId;
    private LocalDate periodStart;
    private LocalDate periodEnd;
    private String assessor;
    private String remark;
}

/**
 * 供应商绩效评估汇总看板 DTO
 */
@Data
class SupplierPerformanceDashboard {
    /**
     * 总供应商数
     */
    private Integer totalSuppliers;
    
    /**
     * A级供应商数
     */
    private Integer levelACount;
    
    /**
     * B级供应商数
     */
    private Integer levelBCount;
    
    /**
     * C级供应商数
     */
    private Integer levelCCount;
    
    /**
     * D级供应商数
     */
    private Integer levelDCount;
    
    /**
     * 平均交货及时率
     */
    private BigDecimal avgOnTimeDeliveryRate;
    
    /**
     * 平均质量合格率
     */
    private BigDecimal avgQualityPassRate;
    
    /**
     * 平均价格竞争力评分
     */
    private BigDecimal avgPriceCompetitivenessScore;
    
    /**
     * 平均综合评分
     */
    private BigDecimal avgOverallScore;
    
    /**
     * 待改进供应商列表
     */
    private List<SupplierPerformanceDTO> improvementNeededSuppliers;
}