package com.metawebthree.finance.domain.service;

import com.metawebthree.finance.domain.entity.cost.*;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;

public class CostAccountingDomainService {

    public void allocateResourceToActivity(ResourcePool pool, Activity activity, BigDecimal amount) {
        BigDecimal available = pool.getAvailableAmount();
        if (available.compareTo(amount) < 0) {
            throw new IllegalArgumentException("Resource pool insufficient funds: available " + available + ", required " + amount);
        }
        pool.allocate(amount);
        activity.assignCost(amount, BigDecimal.ONE);
    }

    public BigDecimal calculateActivityRate(Activity activity) {
        if (activity.getDriverQuantity() == null || activity.getDriverQuantity().compareTo(BigDecimal.ZERO) == 0) {
            return BigDecimal.ZERO;
        }
        return activity.getTotalCost().divide(activity.getDriverQuantity(), 4, RoundingMode.HALF_UP);
    }

    public BigDecimal calculateProductCost(List<Activity> activities, BigDecimal driverQuantity) {
        BigDecimal totalCost = BigDecimal.ZERO;
        for (Activity activity : activities) {
            BigDecimal rate = calculateActivityRate(activity);
            totalCost = totalCost.add(rate.multiply(driverQuantity));
        }
        return totalCost;
    }

    public CostVariance analyzeVariance(StandardCost standard, ActualCost actual) {
        CostVariance variance = new CostVariance();
        variance.calculate(
            actual.getProductCode(),
            actual.getProductName(),
            actual.getProductionOrderNo(),
            actual.getCostDate(),
            standard.getStandardMaterialCost(),
            actual.getActualMaterialCost(),
            standard.getStandardLaborCost(),
            actual.getActualLaborCost(),
            standard.getStandardOverheadCost(),
            actual.getActualOverheadCost()
        );
        variance.analyze();
        return variance;
    }

    public BigDecimal calculateMaterialVariance(BigDecimal standardPrice, BigDecimal actualPrice,
                                                  BigDecimal standardQuantity, BigDecimal actualQuantity) {
        BigDecimal priceVariance = actualQuantity.multiply(standardPrice.subtract(actualPrice));
        BigDecimal quantityVariance = standardPrice.multiply(actualQuantity.subtract(standardQuantity));
        return priceVariance.add(quantityVariance);
    }

    public BigDecimal calculateLaborVariance(BigDecimal standardRate, BigDecimal actualRate,
                                              BigDecimal standardHours, BigDecimal actualHours) {
        BigDecimal rateVariance = actualHours.multiply(standardRate.subtract(actualRate));
        BigDecimal efficiencyVariance = standardRate.multiply(actualHours.subtract(standardHours));
        return rateVariance.add(efficiencyVariance);
    }

    public BigDecimal calculateOverheadVariance(BigDecimal standardRate, BigDecimal actualRate,
                                                 BigDecimal standardHours, BigDecimal actualHours) {
        return calculateLaborVariance(standardRate, actualRate, standardHours, actualHours);
    }

    public List<CostVariance> batchAnalyzeVariance(List<StandardCost> standards, List<ActualCost> actuals) {
        List<CostVariance> results = new ArrayList<>();
        for (ActualCost actual : actuals) {
            for (StandardCost standard : standards) {
                if (standard.getProductCode().equals(actual.getProductCode()) && standard.isEffective()) {
                    results.add(analyzeVariance(standard, actual));
                    break;
                }
            }
        }
        return results;
    }
}