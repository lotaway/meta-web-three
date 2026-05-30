package com.metawebthree.finance.application.query.cost;

import com.metawebthree.finance.domain.entity.cost.*;
import com.metawebthree.finance.domain.repository.cost.*;
import org.springframework.stereotype.Service;
import java.time.LocalDate;
import java.util.List;

@Service
public class CostQueryService {

    private final CostCenterRepository costCenterRepository;
    private final CostDriverRepository costDriverRepository;
    private final ResourcePoolRepository resourcePoolRepository;
    private final ActivityRepository activityRepository;
    private final StandardCostRepository standardCostRepository;
    private final ActualCostRepository actualCostRepository;
    private final CostVarianceRepository costVarianceRepository;

    public CostQueryService(CostCenterRepository costCenterRepository,
                            CostDriverRepository costDriverRepository,
                            ResourcePoolRepository resourcePoolRepository,
                            ActivityRepository activityRepository,
                            StandardCostRepository standardCostRepository,
                            ActualCostRepository actualCostRepository,
                            CostVarianceRepository costVarianceRepository) {
        this.costCenterRepository = costCenterRepository;
        this.costDriverRepository = costDriverRepository;
        this.resourcePoolRepository = resourcePoolRepository;
        this.activityRepository = activityRepository;
        this.standardCostRepository = standardCostRepository;
        this.actualCostRepository = actualCostRepository;
        this.costVarianceRepository = costVarianceRepository;
    }

    public List<CostCenter> findAllCostCenters() {
        return costCenterRepository.findAll();
    }

    public CostCenter findCostCenterById(Long id) {
        return costCenterRepository.findById(id);
    }

    public CostCenter findCostCenterByCode(String code) {
        return costCenterRepository.findByCode(code);
    }

    public List<CostCenter> findCostCentersByType(CostCenter.CostCenterType type) {
        return costCenterRepository.findByType(type);
    }

    public List<CostDriver> findAllCostDrivers() {
        return costDriverRepository.findAll();
    }

    public CostDriver findCostDriverById(Long id) {
        return costDriverRepository.findById(id);
    }

    public List<ResourcePool> findAllResourcePools() {
        return resourcePoolRepository.findAll();
    }

    public ResourcePool findResourcePoolById(Long id) {
        return resourcePoolRepository.findById(id);
    }

    public List<ResourcePool> findResourcePoolsByCostCenterId(Long costCenterId) {
        return resourcePoolRepository.findByCostCenterId(costCenterId);
    }

    public List<Activity> findAllActivities() {
        return activityRepository.findAll();
    }

    public Activity findActivityById(Long id) {
        return activityRepository.findById(id);
    }

    public List<Activity> findActivitiesByCostCenterId(Long costCenterId) {
        return activityRepository.findByCostCenterId(costCenterId);
    }

    public List<StandardCost> findAllStandardCosts() {
        return standardCostRepository.findAll();
    }

    public StandardCost findStandardCostById(Long id) {
        return standardCostRepository.findById(id);
    }

    public StandardCost findEffectiveStandardCostByProductCode(String productCode) {
        return standardCostRepository.findEffectiveByProductCode(productCode);
    }

    public List<StandardCost> findStandardCostsByCategory(String category) {
        return standardCostRepository.findByProductCategory(category);
    }

    public List<ActualCost> findAllActualCosts() {
        return actualCostRepository.findAll();
    }

    public ActualCost findActualCostById(Long id) {
        return actualCostRepository.findById(id);
    }

    public List<ActualCost> findActualCostsByProductCode(String productCode) {
        return actualCostRepository.findByProductCode(productCode);
    }

    public List<ActualCost> findActualCostsByCostDateBetween(LocalDate startDate, LocalDate endDate) {
        return actualCostRepository.findByCostDateBetween(startDate, endDate);
    }

    public List<CostVariance> findAllCostVariances() {
        return costVarianceRepository.findAll();
    }

    public CostVariance findCostVarianceById(Long id) {
        return costVarianceRepository.findById(id);
    }

    public List<CostVariance> findCostVariancesByProductCode(String productCode) {
        return costVarianceRepository.findByProductCode(productCode);
    }

    public List<CostVariance> findCostVariancesByVarianceDateBetween(LocalDate startDate, LocalDate endDate) {
        return costVarianceRepository.findByVarianceDateBetween(startDate, endDate);
    }
}