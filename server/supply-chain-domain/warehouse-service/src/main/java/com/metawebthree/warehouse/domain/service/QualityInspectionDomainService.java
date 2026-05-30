package com.metawebthree.warehouse.domain.service;

import com.metawebthree.warehouse.domain.entity.QualityInspection;
import com.metawebthree.warehouse.domain.entity.QualityInspectionItem;
import com.metawebthree.warehouse.domain.entity.QualityStandard;
import com.metawebthree.warehouse.domain.entity.DefectRecord;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.util.List;

/**
 * 质检领域服务
 */
@Service
public class QualityInspectionDomainService {

    /**
     * 根据质检标准计算抽检数量
     */
    public int calculateSampleQuantity(int totalQuantity, QualityStandard standard) {
        if (standard == null || !standard.isSampleInspection()) {
            return totalQuantity;
        }
        
        BigDecimal rate = standard.getSampleRate();
        if (rate == null) {
            rate = BigDecimal.valueOf(100);
        }
        
        double sampleRate = rate.doubleValue() / 100.0;
        return (int) Math.ceil(totalQuantity * sampleRate);
    }

    /**
     * 判断是否合格
     */
    public boolean isQualified(QualityInspection inspection, QualityStandard standard) {
        if (standard == null) {
            return true; // 没有标准默认合格
        }
        
        BigDecimal defectRate = inspection.getDefectRate();
        if (defectRate == null) {
            defectRate = BigDecimal.ZERO;
        }
        
        Integer threshold = standard.getDefectQtyThreshold();
        if (threshold == null) {
            threshold = 0;
        }
        
        // 不良数量阈值判断
        Integer unqualifiedQty = inspection.getUnqualifiedQuantity();
        if (unqualifiedQty != null && unqualifiedQty > threshold) {
            return false;
        }
        
        // 不良率判断
        BigDecimal thresholdRate = BigDecimal.valueOf(threshold)
            .divide(BigDecimal.valueOf(inspection.getTotalQuantity()), 4, BigDecimal.ROUND_HALF_UP)
            .multiply(BigDecimal.valueOf(100));
        
        return defectRate.compareTo(thresholdRate) <= 0;
    }

    /**
     * 判断是否让步接收
     */
    public boolean canConcession(QualityInspection inspection, QualityStandard standard) {
        if (standard == null) {
            return false;
        }
        
        Integer acceptanceQty = standard.getAcceptanceQty();
        if (acceptanceQty == null) {
            acceptanceQty = 0;
        }
        
        Integer unqualifiedQty = inspection.getUnqualifiedQuantity();
        if (unqualifiedQty == null) {
            unqualifiedQty = 0;
        }
        
        // 不良数量在可接受范围内，可以让步接收
        return unqualifiedQty <= acceptanceQty;
    }

    /**
     * 汇总质检结果
     */
    public void summarizeInspectionResult(QualityInspection inspection, List<QualityInspectionItem> items) {
        int totalQualified = 0;
        int totalUnqualified = 0;
        int totalConcession = 0;
        int totalInspected = 0;
        
        for (QualityInspectionItem item : items) {
            totalQualified += item.getQualifiedQuantity() != null ? item.getQualifiedQuantity() : 0;
            totalUnqualified += item.getUnqualifiedQuantity() != null ? item.getUnqualifiedQuantity() : 0;
            totalConcession += item.getConcessionQuantity() != null ? item.getConcessionQuantity() : 0;
            totalInspected += item.getInspectedQuantity() != null ? item.getInspectedQuantity() : 0;
        }
        
        inspection.setQualifiedQuantity(totalQualified);
        inspection.setUnqualifiedQuantity(totalUnqualified);
        inspection.setConcessionQuantity(totalConcession);
        inspection.setInspectedQuantity(totalInspected);
        
        // 计算不良率
        if (totalInspected > 0) {
            BigDecimal rate = BigDecimal.valueOf(totalUnqualified)
                .multiply(BigDecimal.valueOf(100))
                .divide(BigDecimal.valueOf(totalInspected), 2, BigDecimal.ROUND_HALF_UP);
            inspection.setDefectRate(rate);
        }
        
        // 确定最终状态
        if (totalUnqualified == 0 && totalConcession == 0) {
            inspection.setInspectionStatus(QualityInspection.STATUS_PASSED);
        } else if (totalConcession > 0 && totalUnqualified == 0) {
            inspection.setInspectionStatus(QualityInspection.STATUS_CONCESSION);
        } else {
            inspection.setInspectionStatus(QualityInspection.STATUS_FAILED);
        }
    }

    /**
     * 判断不良等级
     */
    public String determineDefectLevel(DefectRecord defect, QualityStandard standard) {
        if (defect.getDefectQuantity() == null || standard == null) {
            return DefectRecord.LEVEL_MINOR;
        }
        
        int defectQty = defect.getDefectQuantity();
        Integer threshold = standard.getDefectQtyThreshold();
        if (threshold == null) {
            threshold = 0;
        }
        
        // 根据不良数量占总量的比例和阈值判断等级
        if (defectQty > threshold * 2) {
            return DefectRecord.LEVEL_CRITICAL;
        } else if (defectQty > threshold) {
            return DefectRecord.LEVEL_MAJOR;
        } else {
            return DefectRecord.LEVEL_MINOR;
        }
    }
}