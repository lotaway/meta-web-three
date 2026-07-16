package com.metawebthree.dom.application;

import com.metawebthree.dom.application.dto.*;
import com.metawebthree.dom.domain.entity.DomOrder;
import com.metawebthree.dom.domain.entity.DomOrderLine;
import com.metawebthree.dom.domain.entity.FulfillmentPlan;
import com.metawebthree.dom.domain.entity.SourcingRule;
import com.metawebthree.dom.domain.repository.DomOrderLineRepository;
import com.metawebthree.dom.domain.repository.DomOrderRepository;
import com.metawebthree.dom.domain.repository.FulfillmentPlanRepository;
import com.metawebthree.dom.domain.repository.SourcingRuleRepository;
import com.metawebthree.dom.domain.service.DomDomainService;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class DomApplicationServiceImpl implements DomApplicationService {

    private final DomDomainService domDomainService;
    private final DomOrderRepository domOrderRepository;
    private final DomOrderLineRepository domOrderLineRepository;
    private final FulfillmentPlanRepository fulfillmentPlanRepository;
    private final SourcingRuleRepository sourcingRuleRepository;

    public DomApplicationServiceImpl(DomDomainService domDomainService,
                                     DomOrderRepository domOrderRepository,
                                     DomOrderLineRepository domOrderLineRepository,
                                     FulfillmentPlanRepository fulfillmentPlanRepository,
                                     SourcingRuleRepository sourcingRuleRepository) {
        this.domDomainService = domDomainService;
        this.domOrderRepository = domOrderRepository;
        this.domOrderLineRepository = domOrderLineRepository;
        this.fulfillmentPlanRepository = fulfillmentPlanRepository;
        this.sourcingRuleRepository = sourcingRuleRepository;
    }

    @Override
    @Transactional
    public DomOrderDTO createDomOrder(CreateDomOrderRequest request) {
        DomOrder order = new DomOrder();
        order.setOriginalOrderNo(request.getOriginalOrderNo());
        order.setCustomerId(request.getCustomerId());
        order.setCustomerName(request.getCustomerName());
        order.setRegion(request.getRegion());
        order.setSourcingStrategy(request.getSourcingStrategy());
        order.setTotalAmount(BigDecimal.ZERO);
        order.setPriority(0);
        order.setCurrency("CNY");

        List<DomOrderLine> lines = request.getItems().stream().map(item -> {
            DomOrderLine line = new DomOrderLine();
            line.setSkuCode(item.getSkuCode());
            line.setSkuName(item.getSkuName());
            line.setQuantity(item.getQuantity());
            line.setUnitPrice(item.getUnitPrice());
            return line;
        }).collect(Collectors.toList());

        BigDecimal total = lines.stream()
                .map(l -> l.getUnitPrice().multiply(BigDecimal.valueOf(l.getQuantity())))
                .reduce(BigDecimal.ZERO, BigDecimal::add);
        order.setTotalAmount(total);

        DomOrder saved = domDomainService.createDomOrder(order, lines);

        List<DomOrderLine> savedLines = domOrderLineRepository.findByDomOrderId(saved.getId());

        String strategy = request.getSourcingStrategy() != null ? request.getSourcingStrategy() : "BALANCED";

        saved = domDomainService.checkAvailability(saved, savedLines);

        if ("SOURCING".equals(saved.getStatus())) {
            savedLines = domOrderLineRepository.findByDomOrderId(saved.getId());
            List<DomOrderLine> sourcedLines = domDomainService.sourceOrder(saved, savedLines, strategy);
            if (sourcedLines.stream().anyMatch(l -> "SOURCED".equals(l.getStatus()))) {
                savedLines = domOrderLineRepository.findByDomOrderId(saved.getId());
                FulfillmentPlan plan = domDomainService.createFulfillmentPlan(saved, sourcedLines);
            }
        }

        return toDomOrderDTO(domOrderRepository.findById(saved.getId()).orElse(saved),
                domOrderLineRepository.findByDomOrderId(saved.getId()));
    }

    @Override
    public DomOrderDTO getDomOrder(Long id) {
        DomOrder order = domOrderRepository.findById(id).orElse(null);
        if (order == null) {
            return null;
        }
        List<DomOrderLine> lines = domOrderLineRepository.findByDomOrderId(id);
        return toDomOrderDTO(order, lines);
    }

    @Override
    public DomOrderDTO getDomOrderByNo(String domOrderNo) {
        DomOrder order = domOrderRepository.findByDomOrderNo(domOrderNo).orElse(null);
        if (order == null) {
            return null;
        }
        List<DomOrderLine> lines = domOrderLineRepository.findByDomOrderId(order.getId());
        return toDomOrderDTO(order, lines);
    }

    @Override
    public List<DomOrderDTO> listDomOrders(DomQueryParam param) {
        List<DomOrder> orders;
        if (param.getStatus() != null && !param.getStatus().isEmpty()) {
            orders = domOrderRepository.findByStatus(param.getStatus());
        } else {
            orders = domOrderRepository.findAll();
        }
        return orders.stream()
                .map(order -> {
                    List<DomOrderLine> lines = domOrderLineRepository.findByDomOrderId(order.getId());
                    return toDomOrderDTO(order, lines);
                })
                .collect(Collectors.toList());
    }

    @Override
    @Transactional
    public DomOrderDTO checkAvailability(Long orderId) {
        DomOrder order = domOrderRepository.findById(orderId)
                .orElseThrow(() -> new IllegalArgumentException("DOM order not found: " + orderId));
        List<DomOrderLine> lines = domOrderLineRepository.findByDomOrderId(orderId);
        DomOrder updated = domDomainService.checkAvailability(order, lines);
        List<DomOrderLine> updatedLines = domOrderLineRepository.findByDomOrderId(orderId);
        return toDomOrderDTO(updated, updatedLines);
    }

    @Override
    @Transactional
    public DomOrderDTO sourceOrder(Long orderId) {
        DomOrder order = domOrderRepository.findById(orderId)
                .orElseThrow(() -> new IllegalArgumentException("DOM order not found: " + orderId));
        List<DomOrderLine> lines = domOrderLineRepository.findByDomOrderId(orderId);
        String strategy = order.getSourcingStrategy() != null ? order.getSourcingStrategy() : "BALANCED";
        List<DomOrderLine> sourcedLines = domDomainService.sourceOrder(order, lines, strategy);

        if (sourcedLines.stream().anyMatch(l -> "SOURCED".equals(l.getStatus()))) {
            List<DomOrderLine> updatedLines = domOrderLineRepository.findByDomOrderId(orderId);
            domDomainService.createFulfillmentPlan(order, updatedLines);
        }

        List<DomOrderLine> finalLines = domOrderLineRepository.findByDomOrderId(orderId);
        return toDomOrderDTO(order, finalLines);
    }

    @Override
    @Transactional
    public FulfillmentPlanDTO approveFulfillment(Long orderId) {
        DomOrder order = domOrderRepository.findById(orderId)
                .orElseThrow(() -> new IllegalArgumentException("DOM order not found: " + orderId));
        FulfillmentPlan plan = fulfillmentPlanRepository.findByDomOrderId(orderId)
                .orElseThrow(() -> new IllegalArgumentException("Fulfillment plan not found for order: " + orderId));
        FulfillmentPlan approved = domDomainService.approveFulfillmentPlan(plan.getId());
        return toFulfillmentPlanDTO(approved);
    }

    @Override
    @Transactional
    public DomOrderDTO cancelDomOrder(Long orderId) {
        DomOrder order = domDomainService.cancelDomOrder(orderId);
        List<DomOrderLine> lines = domOrderLineRepository.findByDomOrderId(orderId);
        return toDomOrderDTO(order, lines);
    }

    @Override
    public List<SourcingRuleDTO> getSourcingRules() {
        return sourcingRuleRepository.findAll().stream()
                .map(this::toSourcingRuleDTO)
                .collect(Collectors.toList());
    }

    @Override
    @Transactional
    public SourcingRuleDTO updateSourcingRule(SourcingRuleDTO dto) {
        SourcingRule rule = sourcingRuleRepository.findById(dto.getId())
                .orElseThrow(() -> new IllegalArgumentException("Sourcing rule not found: " + dto.getId()));
        rule.setRuleName(dto.getRuleName());
        rule.setRuleType(dto.getRuleType());
        rule.setPriority(dto.getPriority());
        rule.setWarehouseIds(dto.getWarehouseIds());
        rule.setRegion(dto.getRegion());
        rule.setEnabled(dto.getEnabled());
        SourcingRule saved = sourcingRuleRepository.save(rule);
        return toSourcingRuleDTO(saved);
    }

    @Override
    @Transactional
    public SourcingRuleDTO createSourcingRule(SourcingRuleDTO dto) {
        SourcingRule rule = new SourcingRule();
        rule.setRuleName(dto.getRuleName());
        rule.setRuleType(dto.getRuleType());
        rule.setPriority(dto.getPriority());
        rule.setWarehouseIds(dto.getWarehouseIds());
        rule.setRegion(dto.getRegion());
        rule.setEnabled(dto.getEnabled() != null ? dto.getEnabled() : true);
        SourcingRule saved = sourcingRuleRepository.save(rule);
        return toSourcingRuleDTO(saved);
    }

    @Override
    @Transactional
    public void deleteSourcingRule(Long id) {
        SourcingRule rule = sourcingRuleRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Sourcing rule not found: " + id));
        sourcingRuleRepository.delete(rule);
    }

    private DomOrderDTO toDomOrderDTO(DomOrder order, List<DomOrderLine> lines) {
        if (order == null) {
            return null;
        }
        DomOrderDTO dto = new DomOrderDTO();
        dto.setId(order.getId());
        dto.setDomOrderNo(order.getDomOrderNo());
        dto.setOriginalOrderNo(order.getOriginalOrderNo());
        dto.setCustomerId(order.getCustomerId());
        dto.setCustomerName(order.getCustomerName());
        dto.setStatus(order.getStatus());
        dto.setTotalAmount(order.getTotalAmount());
        dto.setCurrency(order.getCurrency());
        dto.setPriority(order.getPriority());
        dto.setSourcingStrategy(order.getSourcingStrategy());
        dto.setRegion(order.getRegion());
        dto.setCreatedAt(order.getCreatedAt());
        dto.setUpdatedAt(order.getUpdatedAt());
        if (lines != null) {
            dto.setLines(lines.stream().map(this::toDomOrderLineDTO).collect(Collectors.toList()));
        }
        return dto;
    }

    private DomOrderLineDTO toDomOrderLineDTO(DomOrderLine line) {
        if (line == null) {
            return null;
        }
        DomOrderLineDTO dto = new DomOrderLineDTO();
        dto.setId(line.getId());
        dto.setDomOrderId(line.getDomOrderId());
        dto.setSkuCode(line.getSkuCode());
        dto.setSkuName(line.getSkuName());
        dto.setQuantity(line.getQuantity());
        dto.setFulfilledQuantity(line.getFulfilledQuantity());
        dto.setWarehouseId(line.getWarehouseId());
        dto.setWarehouseName(line.getWarehouseName());
        dto.setUnitPrice(line.getUnitPrice());
        dto.setStatus(line.getStatus());
        dto.setCreatedAt(line.getCreatedAt());
        return dto;
    }

    private FulfillmentPlanDTO toFulfillmentPlanDTO(FulfillmentPlan plan) {
        if (plan == null) {
            return null;
        }
        FulfillmentPlanDTO dto = new FulfillmentPlanDTO();
        dto.setId(plan.getId());
        dto.setDomOrderId(plan.getDomOrderId());
        dto.setDomOrderNo(plan.getDomOrderNo());
        dto.setTotalLines(plan.getTotalLines());
        dto.setFulfilledLines(plan.getFulfilledLines());
        dto.setPartiallyFulfilledLines(plan.getPartiallyFulfilledLines());
        dto.setUnfulfilledLines(plan.getUnfulfilledLines());
        dto.setStatus(plan.getStatus());
        dto.setCreatedAt(plan.getCreatedAt());
        dto.setUpdatedAt(plan.getUpdatedAt());
        return dto;
    }

    private SourcingRuleDTO toSourcingRuleDTO(SourcingRule rule) {
        if (rule == null) {
            return null;
        }
        SourcingRuleDTO dto = new SourcingRuleDTO();
        dto.setId(rule.getId());
        dto.setRuleName(rule.getRuleName());
        dto.setRuleType(rule.getRuleType());
        dto.setPriority(rule.getPriority());
        dto.setWarehouseIds(rule.getWarehouseIds());
        dto.setRegion(rule.getRegion());
        dto.setEnabled(rule.getEnabled());
        dto.setCreatedAt(rule.getCreatedAt());
        dto.setUpdatedAt(rule.getUpdatedAt());
        return dto;
    }
}
