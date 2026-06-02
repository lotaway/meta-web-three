package com.metawebthree.finance.interfaces.controller;

import com.metawebthree.finance.application.command.cost.CostCommandService;
import com.metawebthree.finance.application.command.cost.dto.*;
import com.metawebthree.finance.application.query.cost.CostQueryService;
import com.metawebthree.finance.domain.entity.cost.*;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/finance/cost")
public class CostAccountingController {

    private final CostCommandService commandService;
    private final CostQueryService queryService;

    public CostAccountingController(CostCommandService commandService, CostQueryService queryService) {
        this.commandService = commandService;
        this.queryService = queryService;
    }

    // ==================== Cost Center ====================

    @PostMapping("/cost-center")
    public ResponseEntity<Map<String, Object>> createCostCenter(@RequestBody CostCenterCreateCommand command) {
        CostCenter costCenter = commandService.createCostCenter(command);
        return ResponseEntity.ok(Map.of("id", costCenter.getId(), "success", true));
    }

    @GetMapping("/cost-center/{id}")
    public ResponseEntity<CostCenter> getCostCenter(@PathVariable Long id) {
        CostCenter costCenter = queryService.findCostCenterById(id);
        return costCenter != null ? ResponseEntity.ok(costCenter) : ResponseEntity.notFound().build();
    }

    @GetMapping("/cost-center/code/{code}")
    public ResponseEntity<CostCenter> getCostCenterByCode(@PathVariable String code) {
        CostCenter costCenter = queryService.findCostCenterByCode(code);
        return costCenter != null ? ResponseEntity.ok(costCenter) : ResponseEntity.notFound().build();
    }

    @GetMapping("/cost-center/list")
    public ResponseEntity<List<CostCenter>> listCostCenters(
            @RequestParam(required = false) String type) {
        List<CostCenter> costCenters;
        if (type != null) {
            costCenters = queryService.findCostCentersByType(CostCenter.CostCenterType.valueOf(type));
        } else {
            costCenters = queryService.findAllCostCenters();
        }
        return ResponseEntity.ok(costCenters);
    }

    // ==================== Cost Driver ====================

    @PostMapping("/cost-driver")
    public ResponseEntity<Map<String, Object>> createCostDriver(@RequestBody CostDriverCreateCommand command) {
        CostDriver costDriver = commandService.createCostDriver(command);
        return ResponseEntity.ok(Map.of("id", costDriver.getId(), "success", true));
    }

    @GetMapping("/cost-driver/{id}")
    public ResponseEntity<CostDriver> getCostDriver(@PathVariable Long id) {
        CostDriver costDriver = queryService.findCostDriverById(id);
        return costDriver != null ? ResponseEntity.ok(costDriver) : ResponseEntity.notFound().build();
    }

    @GetMapping("/cost-driver/list")
    public ResponseEntity<List<CostDriver>> listCostDrivers() {
        return ResponseEntity.ok(queryService.findAllCostDrivers());
    }

    // ==================== Resource Pool ====================

    @PostMapping("/resource-pool")
    public ResponseEntity<Map<String, Object>> createResourcePool(@RequestBody ResourcePoolCreateCommand command) {
        ResourcePool resourcePool = commandService.createResourcePool(command);
        return ResponseEntity.ok(Map.of("id", resourcePool.getId(), "success", true));
    }

    @GetMapping("/resource-pool/{id}")
    public ResponseEntity<ResourcePool> getResourcePool(@PathVariable Long id) {
        ResourcePool resourcePool = queryService.findResourcePoolById(id);
        return resourcePool != null ? ResponseEntity.ok(resourcePool) : ResponseEntity.notFound().build();
    }

    @GetMapping("/resource-pool/cost-center/{costCenterId}")
    public ResponseEntity<List<ResourcePool>> listResourcePoolsByCostCenter(@PathVariable Long costCenterId) {
        return ResponseEntity.ok(queryService.findResourcePoolsByCostCenterId(costCenterId));
    }

    @GetMapping("/resource-pool/list")
    public ResponseEntity<List<ResourcePool>> listResourcePools() {
        return ResponseEntity.ok(queryService.findAllResourcePools());
    }

    // ==================== Activity ====================

    @GetMapping("/activity/{id}")
    public ResponseEntity<Activity> getActivity(@PathVariable Long id) {
        Activity activity = queryService.findActivityById(id);
        return activity != null ? ResponseEntity.ok(activity) : ResponseEntity.notFound().build();
    }

    @GetMapping("/activity/cost-center/{costCenterId}")
    public ResponseEntity<List<Activity>> listActivitiesByCostCenter(@PathVariable Long costCenterId) {
        return ResponseEntity.ok(queryService.findActivitiesByCostCenterId(costCenterId));
    }

    @GetMapping("/activity/list")
    public ResponseEntity<List<Activity>> listActivities() {
        return ResponseEntity.ok(queryService.findAllActivities());
    }

    // ==================== Standard Cost ====================

    @PostMapping("/standard-cost")
    public ResponseEntity<Map<String, Object>> createStandardCost(@RequestBody StandardCostCreateCommand command) {
        StandardCost standardCost = commandService.createStandardCost(command);
        return ResponseEntity.ok(Map.of("id", standardCost.getId(), "success", true));
    }

    @GetMapping("/standard-cost/{id}")
    public ResponseEntity<StandardCost> getStandardCost(@PathVariable Long id) {
        StandardCost standardCost = queryService.findStandardCostById(id);
        return standardCost != null ? ResponseEntity.ok(standardCost) : ResponseEntity.notFound().build();
    }

    @GetMapping("/standard-cost/product/{productCode}")
    public ResponseEntity<StandardCost> getEffectiveStandardCost(@PathVariable String productCode) {
        StandardCost standardCost = queryService.findEffectiveStandardCostByProductCode(productCode);
        return standardCost != null ? ResponseEntity.ok(standardCost) : ResponseEntity.notFound().build();
    }

    @GetMapping("/standard-cost/category/{category}")
    public ResponseEntity<List<StandardCost>> listStandardCostsByCategory(@PathVariable String category) {
        return ResponseEntity.ok(queryService.findStandardCostsByCategory(category));
    }

    @GetMapping("/standard-cost/list")
    public ResponseEntity<List<StandardCost>> listStandardCosts() {
        return ResponseEntity.ok(queryService.findAllStandardCosts());
    }

    // ==================== Actual Cost ====================

    @PostMapping("/actual-cost")
    public ResponseEntity<Map<String, Object>> createActualCost(@RequestBody ActualCostCreateCommand command) {
        ActualCost actualCost = commandService.createActualCost(command);
        return ResponseEntity.ok(Map.of("id", actualCost.getId(), "success", true));
    }

    @GetMapping("/actual-cost/{id}")
    public ResponseEntity<ActualCost> getActualCost(@PathVariable Long id) {
        ActualCost actualCost = queryService.findActualCostById(id);
        return actualCost != null ? ResponseEntity.ok(actualCost) : ResponseEntity.notFound().build();
    }

    @GetMapping("/actual-cost/product/{productCode}")
    public ResponseEntity<List<ActualCost>> listActualCostsByProduct(@PathVariable String productCode) {
        return ResponseEntity.ok(queryService.findActualCostsByProductCode(productCode));
    }

    @GetMapping("/actual-cost/period")
    public ResponseEntity<List<ActualCost>> listActualCostsByPeriod(
            @RequestParam String startDate,
            @RequestParam String endDate) {
        return ResponseEntity.ok(queryService.findActualCostsByCostDateBetween(
                LocalDate.parse(startDate), LocalDate.parse(endDate)));
    }

    @GetMapping("/actual-cost/list")
    public ResponseEntity<List<ActualCost>> listActualCosts() {
        return ResponseEntity.ok(queryService.findAllActualCosts());
    }

    // ==================== Cost Variance ====================

    @GetMapping("/variance/{id}")
    public ResponseEntity<CostVariance> getCostVariance(@PathVariable Long id) {
        CostVariance costVariance = queryService.findCostVarianceById(id);
        return costVariance != null ? ResponseEntity.ok(costVariance) : ResponseEntity.notFound().build();
    }

    @GetMapping("/variance/product/{productCode}")
    public ResponseEntity<List<CostVariance>> listCostVariancesByProduct(@PathVariable String productCode) {
        return ResponseEntity.ok(queryService.findCostVariancesByProductCode(productCode));
    }

    @GetMapping("/variance/period")
    public ResponseEntity<List<CostVariance>> listCostVariancesByPeriod(
            @RequestParam String startDate,
            @RequestParam String endDate) {
        return ResponseEntity.ok(queryService.findCostVariancesByVarianceDateBetween(
                LocalDate.parse(startDate), LocalDate.parse(endDate)));
    }

    @GetMapping("/variance/list")
    public ResponseEntity<List<CostVariance>> listCostVariances() {
        return ResponseEntity.ok(queryService.findAllCostVariances());
    }

    // ==================== Variance Analysis ====================

    @PostMapping("/variance/analyze")
    public ResponseEntity<Map<String, Object>> analyzeVariance(@RequestParam String productCode) {
        CostVariance variance = commandService.analyzeVariance(productCode);
        return ResponseEntity.ok(Map.of("variance", variance, "success", true));
    }
}