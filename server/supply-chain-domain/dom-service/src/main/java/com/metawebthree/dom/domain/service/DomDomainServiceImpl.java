package com.metawebthree.dom.domain.service;

import com.metawebthree.dom.domain.entity.DomOrder;
import com.metawebthree.dom.domain.entity.DomOrderLine;
import com.metawebthree.dom.domain.entity.FulfillmentPlan;
import com.metawebthree.dom.domain.repository.DomOrderLineRepository;
import com.metawebthree.dom.domain.repository.DomOrderRepository;
import com.metawebthree.dom.domain.repository.FulfillmentPlanRepository;
import com.metawebthree.dom.infrastructure.event.DomDomainEventPublisher;
import com.metawebthree.dom.infrastructure.rpc.InventoryServiceClient;
import com.metawebthree.dom.infrastructure.rpc.WarehouseServiceClient;
import com.metawebthree.dom.infrastructure.rpc.WarehouseServiceClient.WarehouseInfo;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Service
public class DomDomainServiceImpl implements DomDomainService {

    private static final AtomicLong SEQ_COUNTER = new AtomicLong(0);
    private static final List<Long> ALL_WAREHOUSES = Arrays.asList(1L, 2L, 3L);

    private final DomOrderRepository domOrderRepository;
    private final DomOrderLineRepository domOrderLineRepository;
    private final FulfillmentPlanRepository fulfillmentPlanRepository;
    private final InventoryServiceClient inventoryServiceClient;
    private final WarehouseServiceClient warehouseServiceClient;
    private final DomDomainEventPublisher eventPublisher;

    public DomDomainServiceImpl(DomOrderRepository domOrderRepository,
                                DomOrderLineRepository domOrderLineRepository,
                                FulfillmentPlanRepository fulfillmentPlanRepository,
                                InventoryServiceClient inventoryServiceClient,
                                WarehouseServiceClient warehouseServiceClient,
                                DomDomainEventPublisher eventPublisher) {
        this.domOrderRepository = domOrderRepository;
        this.domOrderLineRepository = domOrderLineRepository;
        this.fulfillmentPlanRepository = fulfillmentPlanRepository;
        this.inventoryServiceClient = inventoryServiceClient;
        this.warehouseServiceClient = warehouseServiceClient;
        this.eventPublisher = eventPublisher;
    }

    @Override
    @Transactional
    public DomOrder createDomOrder(DomOrder order, List<DomOrderLine> lines) {
        String domOrderNo = generateDomOrderNo();
        order.setDomOrderNo(domOrderNo);
        order.setStatus("PENDING");
        order.setCreatedAt(LocalDateTime.now());
        order.setUpdatedAt(LocalDateTime.now());
        order.setVersion(0);
        DomOrder saved = domOrderRepository.save(order);

        for (DomOrderLine line : lines) {
            line.setDomOrderId(saved.getId());
            line.setStatus("PENDING");
            line.setFulfilledQuantity(0);
            line.setCreatedAt(LocalDateTime.now());
            domOrderLineRepository.save(line);
        }

        eventPublisher.publishDomOrderCreated(saved);
        return saved;
    }

    @Override
    @Transactional
    public DomOrder checkAvailability(DomOrder order, List<DomOrderLine> lines) {
        boolean allPass = true;
        for (DomOrderLine line : lines) {
            int totalAvailable = 0;
            for (Long warehouseId : ALL_WAREHOUSES) {
                Integer qty = inventoryServiceClient.checkInventory(line.getSkuCode(), warehouseId);
                totalAvailable += (qty != null ? qty : 0);
            }
            if (totalAvailable >= line.getQuantity()) {
                line.setStatus("ATP_PASS");
            } else {
                line.setStatus("ATP_FAIL");
                allPass = false;
            }
            domOrderLineRepository.save(line);
        }

        if (allPass) {
            order.setStatus("SOURCING");
        } else {
            order.setStatus("ATP_FAILED");
        }
        order.setUpdatedAt(LocalDateTime.now());
        return domOrderRepository.save(order);
    }

    @Override
    @Transactional
    public List<DomOrderLine> sourceOrder(DomOrder order, List<DomOrderLine> lines, String strategy) {
        String region = order.getRegion();
        List<DomOrderLine> sourcedLines = new ArrayList<>();

        for (DomOrderLine line : lines) {
            if (!"ATP_PASS".equals(line.getStatus())) {
                continue;
            }

            List<WarehouseScore> scored = new ArrayList<>();
            for (Long warehouseId : ALL_WAREHOUSES) {
                Integer available = inventoryServiceClient.checkInventory(line.getSkuCode(), warehouseId);
                if (available == null || available < line.getQuantity()) {
                    continue;
                }
                Double distance = warehouseServiceClient.getWarehouseDistance(region, warehouseId);
                WarehouseInfo info = warehouseServiceClient.getWarehouse(warehouseId);
                double shippingCost = distance * 0.5;
                double handlingCost = 10.0;
                double totalCost = shippingCost + handlingCost;

                double distanceScore = 0.0;
                double costScore = 0.0;

                if ("NEAREST_WAREHOUSE".equals(strategy)) {
                    distanceScore = 100.0 / (distance + 1);
                } else if ("LOWEST_COST".equals(strategy)) {
                    costScore = 1000.0 / (totalCost + 1);
                } else if ("BALANCED".equals(strategy)) {
                    distanceScore = 100.0 / (distance + 1);
                    costScore = 1000.0 / (totalCost + 1);
                }

                double finalScore;
                if ("NEAREST_WAREHOUSE".equals(strategy)) {
                    finalScore = distanceScore;
                } else if ("LOWEST_COST".equals(strategy)) {
                    finalScore = costScore;
                } else {
                    finalScore = 0.5 * distanceScore + 0.5 * costScore;
                }

                scored.add(new WarehouseScore(warehouseId, info != null ? info.getName() : "WH-" + warehouseId, finalScore, totalCost, distance));
            }

            if (scored.isEmpty()) {
                DomOrderLine failedLine = new DomOrderLine();
                failedLine.setId(line.getId());
                failedLine.setSkuCode(line.getSkuCode());
                failedLine.setSkuName(line.getSkuName());
                failedLine.setQuantity(line.getQuantity());
                failedLine.setStatus("SOURCING_FAILED");
                sourcedLines.add(failedLine);
                continue;
            }

            scored.sort((a, b) -> Double.compare(b.score, a.score));
            WarehouseScore best = scored.get(0);

            line.setWarehouseId(best.warehouseId);
            line.setWarehouseName(best.warehouseName);
            line.setStatus("SOURCED");
            domOrderLineRepository.save(line);
            sourcedLines.add(line);
        }

        boolean allSourced = sourcedLines.stream().allMatch(l -> "SOURCED".equals(l.getStatus()));
        if (allSourced) {
            order.setStatus("SOURCING_COMPLETED");
        } else {
            order.setStatus("ATP_FAILED");
        }
        order.setUpdatedAt(LocalDateTime.now());
        domOrderRepository.save(order);

        return sourcedLines;
    }

    @Override
    @Transactional
    public FulfillmentPlan createFulfillmentPlan(DomOrder order, List<DomOrderLine> sourcedLines) {
        FulfillmentPlan plan = new FulfillmentPlan();
        plan.setDomOrderId(order.getId());
        plan.setDomOrderNo(order.getDomOrderNo());
        plan.setTotalLines(sourcedLines.size());

        long fulfilled = sourcedLines.stream().filter(l -> "SOURCED".equals(l.getStatus())).count();
        long unfulfilled = sourcedLines.stream().filter(l -> "SOURCING_FAILED".equals(l.getStatus())).count();

        plan.setFulfilledLines((int) fulfilled);
        plan.setUnfulfilledLines((int) unfulfilled);
        plan.setPartiallyFulfilledLines(0);
        plan.setStatus("DRAFT");
        plan.setCreatedAt(LocalDateTime.now());
        plan.setUpdatedAt(LocalDateTime.now());

        FulfillmentPlan saved = fulfillmentPlanRepository.save(plan);

        eventPublisher.publishDomOrderSourced(order, saved);
        return saved;
    }

    @Override
    @Transactional
    public FulfillmentPlan approveFulfillmentPlan(Long planId) {
        FulfillmentPlan plan = fulfillmentPlanRepository.findById(planId)
                .orElseThrow(() -> new IllegalArgumentException("Fulfillment plan not found: " + planId));

        if (!"DRAFT".equals(plan.getStatus())) {
            throw new IllegalStateException("Plan is not in DRAFT status, current: " + plan.getStatus());
        }

        plan.setStatus("APPROVED");
        plan.setUpdatedAt(LocalDateTime.now());
        FulfillmentPlan saved = fulfillmentPlanRepository.save(plan);

        DomOrder order = domOrderRepository.findById(plan.getDomOrderId())
                .orElseThrow(() -> new IllegalArgumentException("DOM order not found: " + plan.getDomOrderId()));
        order.setStatus("FULFILLED");
        order.setUpdatedAt(LocalDateTime.now());
        domOrderRepository.save(order);

        eventPublisher.publishDomOrderFulfilled(order, saved);
        return saved;
    }

    @Override
    @Transactional
    public DomOrder cancelDomOrder(Long orderId) {
        DomOrder order = domOrderRepository.findById(orderId)
                .orElseThrow(() -> new IllegalArgumentException("DOM order not found: " + orderId));

        if ("CANCELLED".equals(order.getStatus()) || "FULFILLED".equals(order.getStatus())) {
            throw new IllegalStateException("Cannot cancel order in status: " + order.getStatus());
        }

        order.setStatus("CANCELLED");
        order.setUpdatedAt(LocalDateTime.now());
        DomOrder saved = domOrderRepository.save(order);

        List<DomOrderLine> lines = domOrderLineRepository.findByDomOrderId(orderId);
        for (DomOrderLine line : lines) {
            line.setStatus("CANCELLED");
            domOrderLineRepository.save(line);
        }

        return saved;
    }

    private String generateDomOrderNo() {
        String datePart = LocalDate.now().format(DateTimeFormatter.ofPattern("yyyyMMdd"));
        long seq = SEQ_COUNTER.incrementAndGet() % 1000000;
        return "DOM" + datePart + String.format("%06d", seq);
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
