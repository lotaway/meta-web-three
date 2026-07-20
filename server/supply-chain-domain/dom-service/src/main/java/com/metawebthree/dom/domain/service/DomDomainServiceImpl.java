package com.metawebthree.dom.domain.service;

import com.metawebthree.dom.domain.entity.*;
import com.metawebthree.dom.domain.repository.DomOrderLineRepository;
import com.metawebthree.dom.domain.repository.DomOrderRepository;
import com.metawebthree.dom.domain.repository.FulfillmentPlanRepository;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

public class DomDomainServiceImpl implements DomDomainService {

    private final DomOrderRepository domOrderRepository;
    private final DomOrderLineRepository domOrderLineRepository;
    private final FulfillmentPlanRepository fulfillmentPlanRepository;
    private final InventoryServiceClient inventoryServiceClient;
    private final WarehouseServiceClient warehouseServiceClient;
    private final DomDomainEventPublisher eventPublisher;
    private final DomSequenceGenerator sequenceGenerator;
    private final DomSourcingProperties sourcingProperties;

    public DomDomainServiceImpl(DomOrderRepository domOrderRepository,
                                DomOrderLineRepository domOrderLineRepository,
                                FulfillmentPlanRepository fulfillmentPlanRepository,
                                InventoryServiceClient inventoryServiceClient,
                                WarehouseServiceClient warehouseServiceClient,
                                DomDomainEventPublisher eventPublisher,
                                DomSequenceGenerator sequenceGenerator,
                                DomSourcingProperties sourcingProperties) {
        this.domOrderRepository = domOrderRepository;
        this.domOrderLineRepository = domOrderLineRepository;
        this.fulfillmentPlanRepository = fulfillmentPlanRepository;
        this.inventoryServiceClient = inventoryServiceClient;
        this.warehouseServiceClient = warehouseServiceClient;
        this.eventPublisher = eventPublisher;
        this.sequenceGenerator = sequenceGenerator;
        this.sourcingProperties = sourcingProperties;
    }

    @Override
    public DomOrder createDomOrder(DomOrder order, List<DomOrderLine> lines) {
        String domOrderNo = sequenceGenerator.generateDomOrderNo();
        order.setDomOrderNo(domOrderNo);
        order.setStatus(DomOrderStatus.PENDING);
        order.setCreatedAt(LocalDateTime.now());
        order.setUpdatedAt(LocalDateTime.now());
        order.setVersion(0);
        return order;
    }

    @Override
    public void saveDomOrder(DomOrder order, List<DomOrderLine> lines) {
        DomOrder saved = domOrderRepository.save(order);
        for (DomOrderLine line : lines) {
            line.setDomOrderId(saved.getId());
            line.setStatus(DomOrderLineStatus.PENDING);
            line.setFulfilledQuantity(0);
            line.setCreatedAt(LocalDateTime.now());
            domOrderLineRepository.save(line);
        }
        eventPublisher.publishDomOrderCreated(saved);
    }

    @Override
    public boolean checkAvailability(DomOrder order, List<DomOrderLine> lines) {
        boolean allPass = true;
        for (DomOrderLine line : lines) {
            int totalAvailable = sumInventoryAcrossWarehouses(line.getSkuCode());
            line.setStatus(totalAvailable >= line.getQuantity()
                    ? DomOrderLineStatus.ATP_PASS
                    : DomOrderLineStatus.ATP_FAIL);
            if (line.getStatus() == DomOrderLineStatus.ATP_FAIL) {
                allPass = false;
            }
        }
        return allPass;
    }

    private int sumInventoryAcrossWarehouses(String skuCode) {
        int total = 0;
        for (Long warehouseId : sourcingProperties.getWarehouseIds()) {
            Integer qty = inventoryServiceClient.checkInventory(skuCode, warehouseId);
            total += (qty != null ? qty : 0);
        }
        return total;
    }

    @Override
    public void saveAvailabilityResult(DomOrder order, List<DomOrderLine> lines, boolean allPass) {
        for (DomOrderLine line : lines) {
            domOrderLineRepository.save(line);
        }
        order.setStatus(allPass ? DomOrderStatus.SOURCING : DomOrderStatus.ATP_FAILED);
        order.setUpdatedAt(LocalDateTime.now());
        domOrderRepository.save(order);
    }

    @Override
    public List<DomOrderLine> sourceOrder(DomOrder order, List<DomOrderLine> lines, SourcingStrategy strategy) {
        List<DomOrderLine> sourcedLines = new ArrayList<>();
        for (DomOrderLine line : lines) {
            if (line.getStatus() != DomOrderLineStatus.ATP_PASS) continue;
            List<WarehouseScore> scored = scoreWarehouses(line, order.getRegion(), strategy);
            try {
                DomOrderLine result = selectBestSource(scored, line);
                sourcedLines.add(result);
            } catch (SourcingFailedException e) {
                DomOrderLine failedLine = new DomOrderLine();
                failedLine.setId(line.getId());
                failedLine.setSkuCode(line.getSkuCode());
                failedLine.setSkuName(line.getSkuName());
                failedLine.setQuantity(line.getQuantity());
                failedLine.setStatus(DomOrderLineStatus.SOURCING_FAILED);
                sourcedLines.add(failedLine);
            }
        }
        return sourcedLines;
    }

    @Override
    public void saveSourcingResult(DomOrder order, List<DomOrderLine> lines) {
        boolean allSourced = lines.stream().allMatch(l -> l.getStatus() == DomOrderLineStatus.SOURCED);
        order.setStatus(allSourced ? DomOrderStatus.SOURCING_COMPLETED : DomOrderStatus.SOURCING_FAILED);
        order.setUpdatedAt(LocalDateTime.now());
        domOrderRepository.save(order);
    }

    private List<WarehouseScore> scoreWarehouses(DomOrderLine line, String region, SourcingStrategy strategy) {
        List<WarehouseScore> scored = new ArrayList<>();
        for (Long warehouseId : sourcingProperties.getWarehouseIds()) {
            Integer available = inventoryServiceClient.checkInventory(line.getSkuCode(), warehouseId);
            if (available == null || available < line.getQuantity()) continue;

            Double distance = warehouseServiceClient.getWarehouseDistance(region, warehouseId);
            WarehouseInfo info = warehouseServiceClient.getWarehouse(warehouseId);
            double totalCost = distance * sourcingProperties.getShippingCostPerKm() + sourcingProperties.getHandlingCostFlat();

            double distanceScore = sourcingProperties.getDistanceScoreFactor() / (distance + 1);
            double costScore = sourcingProperties.getCostScoreFactor() / (totalCost + 1);
            double finalScore = calculateFinalScore(strategy, distanceScore, costScore);

            String name = info != null ? info.getName() : sourcingProperties.getWhNamePrefix() + warehouseId;
            scored.add(new WarehouseScore(warehouseId, name, finalScore, totalCost, distance));
        }
        return scored;
    }

    private double calculateFinalScore(SourcingStrategy strategy, double distanceScore, double costScore) {
        switch (strategy) {
            case NEAREST_WAREHOUSE: return distanceScore;
            case LOWEST_COST: return costScore;
            default: return sourcingProperties.getBalancedDistanceWeight() * distanceScore
                    + sourcingProperties.getBalancedCostWeight() * costScore;
        }
    }

    private DomOrderLine selectBestSource(List<WarehouseScore> scored, DomOrderLine line) {
        if (scored.isEmpty()) {
            throw new SourcingFailedException("No suitable warehouse found for sku: " + line.getSkuCode());
        }
        scored.sort((a, b) -> Double.compare(b.score, a.score));
        WarehouseScore best = scored.get(0);
        line.setWarehouseId(best.warehouseId);
        line.setWarehouseName(best.warehouseName);
        line.setStatus(DomOrderLineStatus.SOURCED);
        return line;
    }

    @Override
    public FulfillmentPlan createFulfillmentPlan(DomOrder order, List<DomOrderLine> sourcedLines) {
        FulfillmentPlan plan = new FulfillmentPlan();
        plan.setDomOrderId(order.getId());
        plan.setDomOrderNo(order.getDomOrderNo());
        plan.setTotalLines(sourcedLines.size());

        long fulfilled = sourcedLines.stream().filter(l -> l.getStatus() == DomOrderLineStatus.SOURCED).count();
        long unfulfilled = sourcedLines.stream().filter(l -> l.getStatus() == DomOrderLineStatus.SOURCING_FAILED).count();

        plan.setFulfilledLines((int) fulfilled);
        plan.setUnfulfilledLines((int) unfulfilled);
        plan.setPartiallyFulfilledLines(0);
        plan.setStatus(FulfillmentPlanStatus.DRAFT);
        plan.setCreatedAt(LocalDateTime.now());
        plan.setUpdatedAt(LocalDateTime.now());
        return plan;
    }

    @Override
    public void saveFulfillmentPlan(DomOrder order, FulfillmentPlan plan) {
        FulfillmentPlan saved = fulfillmentPlanRepository.save(plan);
        eventPublisher.publishDomOrderSourced(order, saved);
    }

    @Override
    public FulfillmentPlan approveFulfillmentPlan(Long planId) {
        FulfillmentPlan plan = fulfillmentPlanRepository.findById(planId)
                .orElseThrow(() -> new IllegalArgumentException("Fulfillment plan not found: " + planId));
        if (plan.getStatus() != FulfillmentPlanStatus.DRAFT) {
            throw new IllegalStateException("Plan is not in DRAFT status, current: " + plan.getStatus());
        }
        plan.setStatus(FulfillmentPlanStatus.APPROVED);
        plan.setUpdatedAt(LocalDateTime.now());
        return plan;
    }

    @Override
    public void saveApprovedPlan(DomOrder order, FulfillmentPlan plan) {
        FulfillmentPlan saved = fulfillmentPlanRepository.save(plan);
        order.setStatus(DomOrderStatus.FULFILLED);
        order.setUpdatedAt(LocalDateTime.now());
        domOrderRepository.save(order);
        eventPublisher.publishDomOrderFulfilled(order, saved);
    }

    @Override
    public DomOrder cancelDomOrder(Long orderId) {
        DomOrder order = domOrderRepository.findById(orderId)
                .orElseThrow(() -> new IllegalArgumentException("DOM order not found: " + orderId));
        if (order.getStatus() == DomOrderStatus.CANCELLED || order.getStatus() == DomOrderStatus.FULFILLED) {
            throw new IllegalStateException("Cannot cancel order in status: " + order.getStatus());
        }
        order.setStatus(DomOrderStatus.CANCELLED);
        order.setUpdatedAt(LocalDateTime.now());
        return order;
    }

    @Override
    public void saveCancelledOrder(DomOrder order) {
        DomOrder saved = domOrderRepository.save(order);
        List<DomOrderLine> lines = domOrderLineRepository.findByDomOrderId(saved.getId());
        for (DomOrderLine line : lines) {
            line.setStatus(DomOrderLineStatus.CANCELLED);
            domOrderLineRepository.save(line);
        }
    }

    static class WarehouseScore {
        final Long warehouseId;
        final String warehouseName;
        final double score;
        final double cost;
        final double distance;

        WarehouseScore(Long warehouseId, String warehouseName, double score, double cost, double distance) {
            this.warehouseId = warehouseId;
            this.warehouseName = warehouseName;
            this.score = score;
            this.cost = cost;
            this.distance = distance;
        }
    }
}
