package com.metawebthree.supplier.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
public class SupplierPerformance {
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
    
    public void calculateOverallScore() {
        if (onTimeDeliveryRate == null || qualityPassRate == null || priceCompetitivenessScore == null) {
            this.overallScore = BigDecimal.ZERO;
            return;
        }
        // 加权平均：交货及时率 40%，质量合格率 40%，价格竞争力 20%
        this.overallScore = onTimeDeliveryRate.multiply(new BigDecimal("0.4"))
            .add(qualityPassRate.multiply(new BigDecimal("0.4")))
            .add(priceCompetitivenessScore.multiply(new BigDecimal("0.2")));
    }
    
    public void determineAssessmentLevel() {
        if (overallScore == null) {
            this.assessmentLevel = "D";
            return;
        }
        
        if (overallScore.compareTo(new BigDecimal("90")) >= 0) {
            this.assessmentLevel = "A";
        } else if (overallScore.compareTo(new BigDecimal("75")) >= 0) {
            this.assessmentLevel = "B";
        } else if (overallScore.compareTo(new BigDecimal("60")) >= 0) {
            this.assessmentLevel = "C";
        } else {
            this.assessmentLevel = "D";
        }
    }
    
    public void evaluate() {
        calculateOverallScore();
        determineAssessmentLevel();
    }
}