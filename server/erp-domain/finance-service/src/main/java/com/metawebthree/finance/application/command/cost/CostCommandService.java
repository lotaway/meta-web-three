package com.metawebthree.finance.application.command.cost;

import com.metawebthree.finance.application.command.cost.dto.*;
import com.metawebthree.finance.domain.entity.cost.*;
import com.metawebthree.finance.domain.repository.cost.*;
import com.metawebthree.finance.domain.service.CostAccountingDomainService;
import org.springframework.stereotype.Service;
import java.math.BigDecimal;

@Service
public class CostCommandService {

    private final CostCenterRepository costCenterRepository;
    private final CostDriverRepository costDriverRepository;
    private final ResourcePoolRepository resourcePoolRepository;
    private final ActivityRepository activityRepository;
    private final StandardCostRepository standardCostRepository;
    private final ActualCostRepository actualCostRepository;
    private final CostVarianceRepository costVarianceRepository;
    private final CostAccountingDomainService domainService;

    public CostCommandService(CostCenterRepository costCenterRepository,
                              CostDriverRepository costDriverRepository,
                              ResourcePoolRepository resourcePoolRepository,
                              ActivityRepository activityRepository,
                              StandardCostRepository standardCostRepository,
                              ActualCostRepository actualCostRepository,
                              CostVarianceRepository costVarianceRepository,
                              CostAccountingDomainService domainService) {
        this.costCenterRepository = costCenterRepository;
        this.costDriverRepository = costDriverRepository;
        this.resourcePoolRepository = resourcePoolRepository;
        this.activityRepository = activityRepository;
        this.standardCostRepository = standardCostRepository;
        this.actualCostRepository = actualCostRepository;
        this.costVarianceRepository = costVarianceRepository;
        this.domainService = domainService;
    }

    public CostCenter createCostCenter(CostCenterCreateCommand command) {
        CostCenter costCenter = new CostCenter();
        costCenter.create(
            command.getCostCenterCode(),
            command.getCostCenterName(),
            CostCenter.CostCenterType.valueOf(command.getType()),
            command.getDepartmentId(),
            command.getDepartmentName(),
            command.getManagerName(),
            command.getBudgetAmount() != null ? command.getBudgetAmount() : BigDecimal.ZERO,
            command.getCreatedBy()
        );
        costCenter.setDescription(command.getDescription());
        return costCenterRepository.save(costCenter);
    }

    public CostDriver createCostDriver(CostDriverCreateCommand command) {
        CostDriver costDriver = new CostDriver();
        costDriver.create(
            command.getDriverCode(),
            command.getDriverName(),
            CostDriver.CostDriverType.valueOf(command.getType()),
            command.getUnit(),
            command.getRate() != null ? command.getRate() : BigDecimal.ZERO,
            command.getDescription()
        );
        return costDriverRepository.save(costDriver);
    }

    public ResourcePool createResourcePool(ResourcePoolCreateCommand command) {
        ResourcePool resourcePool = new ResourcePool();
        resourcePool.create(
            command.getPoolCode(),
            command.getPoolName(),
            command.getCostCenterId(),
            command.getCostCenterName(),
            ResourcePool.ResourcePoolType.valueOf(command.getType()),
            command.getTotalBudget() != null ? command.getTotalBudget() : BigDecimal.ZERO,
            command.getCurrency() != null ? command.getCurrency() : "CNY",
            command.getDescription()
        );
        return resourcePoolRepository.save(resourcePool);
    }

    public StandardCost createStandardCost(StandardCostCreateCommand command) {
        StandardCost standardCost = new StandardCost();
        standardCost.create(
            command.getProductCode(),
            command.getProductName(),
            command.getProductCategory(),
            command.getStandardMaterialCost() != null ? command.getStandardMaterialCost() : BigDecimal.ZERO,
            command.getStandardLaborCost() != null ? command.getStandardLaborCost() : BigDecimal.ZERO,
            command.getStandardOverheadCost() != null ? command.getStandardOverheadCost() : BigDecimal.ZERO,
            command.getStandardQuantity() != null ? command.getStandardQuantity() : BigDecimal.ONE,
            command.getUnit() != null ? command.getUnit() : "PCS",
            command.getEffectiveDate(),
            command.getVersion() != null ? command.getVersion() : "1.0",
            command.getCreatedBy(),
            command.getCurrency() != null ? command.getCurrency() : "CNY"
        );
        standardCost.setRemark(command.getRemark());
        return standardCostRepository.save(standardCost);
    }

    public ActualCost createActualCost(ActualCostCreateCommand command) {
        ActualCost actualCost = new ActualCost();
        actualCost.create(
            command.getProductCode(),
            command.getProductName(),
            command.getProductionOrderNo(),
            command.getCostCenterId(),
            command.getCostCenterName(),
            command.getCostDate(),
            command.getActualMaterialCost() != null ? command.getActualMaterialCost() : BigDecimal.ZERO,
            command.getActualLaborCost() != null ? command.getActualLaborCost() : BigDecimal.ZERO,
            command.getActualOverheadCost() != null ? command.getActualOverheadCost() : BigDecimal.ZERO,
            command.getQuantity() != null ? command.getQuantity() : BigDecimal.ONE,
            command.getUnit() != null ? command.getUnit() : "PCS",
            command.getCostType() != null ? command.getCostType() : "PRODUCTION",
            command.getCreatedBy(),
            command.getCurrency() != null ? command.getCurrency() : "CNY"
        );
        actualCost.setRemark(command.getRemark());
        return actualCostRepository.save(actualCost);
    }

    public CostVariance analyzeVariance(String productCode) {
        StandardCost standard = standardCostRepository.findEffectiveByProductCode(productCode);
        if (standard == null) {
            throw new IllegalArgumentException("No effective standard cost found for product: " + productCode);
        }

        java.time.LocalDate today = java.time.LocalDate.now();
        java.util.List<ActualCost> actuals = actualCostRepository.findByProductCode(productCode);

        if (actuals.isEmpty()) {
            throw new IllegalArgumentException("No actual cost records found for product: " + productCode);
        }

        ActualCost latestActual = actuals.get(actuals.size() - 1);

        return domainService.analyzeVariance(standard, latestActual);
    }
}