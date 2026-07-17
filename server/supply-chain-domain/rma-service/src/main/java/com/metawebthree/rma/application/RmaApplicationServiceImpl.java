package com.metawebthree.rma.application;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.rma.application.dto.*;
import com.metawebthree.rma.application.event.RmaCompletedEvent;
import com.metawebthree.rma.application.event.RmaCreatedEvent;
import com.metawebthree.rma.application.event.RmaDispositionExecutedEvent;
import com.metawebthree.rma.application.event.RmaInspectionCompletedEvent;
import com.metawebthree.rma.domain.entity.*;
import com.metawebthree.rma.domain.repository.*;
import com.metawebthree.rma.domain.service.RmaDomainService;
import com.metawebthree.rma.infrastructure.event.RmaDomainEventPublisher;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class RmaApplicationServiceImpl implements RmaApplicationService {

    private final RmaDomainService rmaDomainService;
    private final RmaOrderRepository rmaOrderRepository;
    private final RmaOrderItemRepository rmaOrderItemRepository;
    private final RmaInspectionRepository rmaInspectionRepository;
    private final RmaDispositionRepository rmaDispositionRepository;
    private final ReturnShippingRepository returnShippingRepository;
    private final RmaDomainEventPublisher eventPublisher;

    public RmaApplicationServiceImpl(RmaDomainService rmaDomainService,
                                     RmaOrderRepository rmaOrderRepository,
                                     RmaOrderItemRepository rmaOrderItemRepository,
                                     RmaInspectionRepository rmaInspectionRepository,
                                     RmaDispositionRepository rmaDispositionRepository,
                                     ReturnShippingRepository returnShippingRepository,
                                     RmaDomainEventPublisher eventPublisher) {
        this.rmaDomainService = rmaDomainService;
        this.rmaOrderRepository = rmaOrderRepository;
        this.rmaOrderItemRepository = rmaOrderItemRepository;
        this.rmaInspectionRepository = rmaInspectionRepository;
        this.rmaDispositionRepository = rmaDispositionRepository;
        this.returnShippingRepository = returnShippingRepository;
        this.eventPublisher = eventPublisher;
    }

    @Override
    @Transactional
    public RmaOrderDTO createRma(CreateRmaRequest request) {
        List<RmaOrderItem> items = request.getItems().stream().map(i -> {
            RmaOrderItem item = new RmaOrderItem();
            item.setSkuCode(i.getSkuCode());
            item.setSkuName(i.getSkuName());
            item.setExpectedQuantity(i.getExpectedQuantity());
            item.setUnitPrice(i.getUnitPrice());
            item.setReasonCode(i.getReasonCode());
            item.setReasonDescription(i.getReasonDescription());
            return item;
        }).collect(Collectors.toList());

        RmaOrder order = rmaDomainService.createRmaOrder(
                request.getOrderNo(),
                request.getCustomerId(),
                request.getCustomerName(),
                request.getContactPhone(),
                request.getReasonCode(),
                request.getReasonDescription(),
                request.getWarehouseId(),
                request.getReturnType(),
                request.getCreatedBy(),
                items
        );

        rmaDomainService.saveRmaOrder(order, items);
        eventPublisher.publish(new RmaCreatedEvent("RMA_CREATED", order.getId(), order.getRmaNo()));

        return toOrderDTO(order);
    }

    @Override
    public RmaOrderDTO getRma(Long id) {
        return rmaOrderRepository.findById(id)
                .map(this::toOrderDTO)
                .orElse(null);
    }

    @Override
    public RmaOrderDTO getRmaByNo(String rmaNo) {
        return rmaDomainService.getRmaOrderByNo(rmaNo)
                .map(this::toOrderDTO)
                .orElse(null);
    }

    @Override
    public IPage<RmaOrderDTO> listRmas(String status, Integer pageNum, Integer pageSize) {
        Page<RmaOrder> page = new Page<>(pageNum, pageSize);
        IPage<RmaOrder> orderPage = rmaOrderRepository.findPage(page, status);
        IPage<RmaOrderDTO> dtoPage = new Page<>(orderPage.getCurrent(), orderPage.getSize(), orderPage.getTotal());
        dtoPage.setRecords(orderPage.getRecords().stream()
                .map(this::toOrderDTO)
                .collect(Collectors.toList()));
        return dtoPage;
    }

    @Override
    @Transactional
    public RmaOrderDTO submitForInspection(Long rmaId) {
        RmaOrder order = rmaDomainService.submitForInspection(rmaId);
        rmaDomainService.saveRmaOrder(order);
        return toOrderDTO(order);
    }

    @Override
    @Transactional
    public RmaOrderDTO recordInspection(Long rmaId, RecordInspectionRequest request) {
        RmaInspection inspection = new RmaInspection();
        inspection.setInspector(request.getInspector());
        inspection.setResult(request.getResult());
        inspection.setConclusion(request.getConclusion());
        inspection.setTotalInspected(request.getTotalInspected());
        inspection.setTotalPassed(request.getTotalPassed());
        inspection.setTotalFailed(request.getTotalFailed());
        inspection.setRemark(request.getRemark());

        RmaOrder order = rmaDomainService.getRmaOrder(rmaId)
                .orElseThrow(() -> new IllegalArgumentException("RMA order not found"));

        RmaInspection saved = rmaDomainService.recordInspection(order, inspection);
        rmaDomainService.saveInspection(saved);
        rmaDomainService.saveRmaOrder(order);

        eventPublisher.publish(new RmaInspectionCompletedEvent("RMA_INSPECTION_COMPLETED", rmaId, order.getRmaNo(),
                saved.getResult(), saved.getTotalPassed()));

        return toOrderDTO(order);
    }

    @Override
    @Transactional
    public RmaOrderDTO makeDisposition(Long rmaId, MakeDispositionRequest request) {
        RmaDisposition disposition = new RmaDisposition();
        disposition.setDispositionType(request.getDispositionType());
        disposition.setDispositionBy(request.getDispositionBy());
        disposition.setRefundAmount(request.getRefundAmount());
        disposition.setReplacementSkuCode(request.getReplacementSkuCode());
        disposition.setReplacementQuantity(request.getReplacementQuantity());
        disposition.setScrapQuantity(request.getScrapQuantity());
        disposition.setScrapReason(request.getScrapReason());
        disposition.setRemark(request.getRemark());

        RmaOrder order = rmaDomainService.getRmaOrder(rmaId)
                .orElseThrow(() -> new IllegalArgumentException("RMA order not found"));

        rmaDomainService.makeDisposition(order, disposition);
        rmaDomainService.saveDisposition(disposition);
        rmaDomainService.saveRmaOrder(order);

        return toOrderDTO(order);
    }

    @Override
    @Transactional
    public RmaOrderDTO executeDisposition(Long rmaId) {
        RmaOrder order = rmaDomainService.executeDisposition(rmaId);
        rmaDomainService.saveRmaOrder(order);

        RmaDisposition disposition = rmaDispositionRepository.findByRmaId(rmaId)
                .orElse(null);

        if (disposition != null) {
            eventPublisher.publish(new RmaDispositionExecutedEvent("RMA_DISPOSITION_EXECUTED", rmaId, order.getRmaNo(),
                    disposition.getDispositionType()));
        }

        return toOrderDTO(order);
    }

    @Override
    @Transactional
    public RmaOrderDTO completeRma(Long rmaId) {
        RmaOrder order = rmaDomainService.completeRmaOrder(rmaId);
        rmaDomainService.saveRmaOrder(order);
        eventPublisher.publish(new RmaCompletedEvent("RMA_COMPLETED", rmaId, order.getRmaNo()));
        return toOrderDTO(order);
    }

    @Override
    @Transactional
    public RmaOrderDTO cancelRma(Long rmaId) {
        RmaOrder order = rmaDomainService.cancelRmaOrder(rmaId);
        rmaDomainService.saveRmaOrder(order);
        return toOrderDTO(order);
    }

    @Override
    public List<?> getRmaTimeline(Long rmaId) {
        List<Object> timeline = new ArrayList<>();
        rmaOrderRepository.findById(rmaId).ifPresent(order -> timeline.add(order));
        rmaInspectionRepository.findByRmaId(rmaId).ifPresent(inspection -> timeline.add(inspection));
        rmaDispositionRepository.findByRmaId(rmaId).ifPresent(disposition -> timeline.add(disposition));
        returnShippingRepository.findByRmaId(rmaId).ifPresent(shipping -> timeline.add(shipping));
        return timeline;
    }

    private RmaOrderDTO toOrderDTO(RmaOrder order) {
        RmaOrderDTO dto = new RmaOrderDTO();
        dto.setId(order.getId());
        dto.setRmaNo(order.getRmaNo());
        dto.setOrderNo(order.getOrderNo());
        dto.setReturnType(order.getReturnType());
        dto.setStatus(order.getStatus() != null ? order.getStatus().name() : null);
        dto.setCustomerId(order.getCustomerId());
        dto.setCustomerName(order.getCustomerName());
        dto.setContactPhone(order.getContactPhone());
        dto.setReasonCode(order.getReasonCode());
        dto.setReasonDescription(order.getReasonDescription());
        dto.setWarehouseId(order.getWarehouseId());
        dto.setTotalQuantity(order.getTotalQuantity());
        dto.setTotalAmount(order.getTotalAmount());
        dto.setCurrency(order.getCurrency());
        dto.setCreatedBy(order.getCreatedBy());
        dto.setCreatedAt(order.getCreatedAt());
        dto.setUpdatedAt(order.getUpdatedAt());

        List<RmaOrderItem> items = rmaOrderItemRepository.findByRmaId(order.getId());
        dto.setItems(items.stream().map(this::toItemDTO).collect(Collectors.toList()));

        rmaInspectionRepository.findByRmaId(order.getId())
                .ifPresent(inspection -> dto.setInspection(toInspectionDTO(inspection)));

        rmaDispositionRepository.findByRmaId(order.getId())
                .ifPresent(disposition -> dto.setDisposition(toDispositionDTO(disposition)));

        returnShippingRepository.findByRmaId(order.getId())
                .ifPresent(shipping -> dto.setShipping(toShippingDTO(shipping)));

        return dto;
    }

    private RmaOrderItemDTO toItemDTO(RmaOrderItem item) {
        RmaOrderItemDTO dto = new RmaOrderItemDTO();
        dto.setId(item.getId());
        dto.setRmaId(item.getRmaId());
        dto.setSkuCode(item.getSkuCode());
        dto.setSkuName(item.getSkuName());
        dto.setExpectedQuantity(item.getExpectedQuantity());
        dto.setInspectedQuantity(item.getInspectedQuantity());
        dto.setAcceptedQuantity(item.getAcceptedQuantity());
        dto.setUnitPrice(item.getUnitPrice());
        dto.setReasonCode(item.getReasonCode());
        dto.setReasonDescription(item.getReasonDescription());
        dto.setCreatedAt(item.getCreatedAt());
        return dto;
    }

    private RmaInspectionDTO toInspectionDTO(RmaInspection inspection) {
        RmaInspectionDTO dto = new RmaInspectionDTO();
        dto.setId(inspection.getId());
        dto.setRmaId(inspection.getRmaId());
        dto.setRmaNo(inspection.getRmaNo());
        dto.setInspector(inspection.getInspector());
        dto.setInspectionDate(inspection.getInspectionDate());
        dto.setResult(inspection.getResult());
        dto.setConclusion(inspection.getConclusion());
        dto.setTotalInspected(inspection.getTotalInspected());
        dto.setTotalPassed(inspection.getTotalPassed());
        dto.setTotalFailed(inspection.getTotalFailed());
        dto.setRemark(inspection.getRemark());
        dto.setCreatedAt(inspection.getCreatedAt());
        dto.setUpdatedAt(inspection.getUpdatedAt());
        return dto;
    }

    private RmaDispositionDTO toDispositionDTO(RmaDisposition disposition) {
        RmaDispositionDTO dto = new RmaDispositionDTO();
        dto.setId(disposition.getId());
        dto.setRmaId(disposition.getRmaId());
        dto.setRmaNo(disposition.getRmaNo());
        dto.setDispositionType(disposition.getDispositionType());
        dto.setRefundAmount(disposition.getRefundAmount());
        dto.setReplacementSkuCode(disposition.getReplacementSkuCode());
        dto.setReplacementQuantity(disposition.getReplacementQuantity());
        dto.setScrapQuantity(disposition.getScrapQuantity());
        dto.setScrapReason(disposition.getScrapReason());
        dto.setDispositionBy(disposition.getDispositionBy());
        dto.setDispositionDate(disposition.getDispositionDate());
        dto.setRemark(disposition.getRemark());
        dto.setCreatedAt(disposition.getCreatedAt());
        dto.setUpdatedAt(disposition.getUpdatedAt());
        return dto;
    }

    private ReturnShippingDTO toShippingDTO(ReturnShipping shipping) {
        ReturnShippingDTO dto = new ReturnShippingDTO();
        dto.setId(shipping.getId());
        dto.setRmaId(shipping.getRmaId());
        dto.setRmaNo(shipping.getRmaNo());
        dto.setCarrier(shipping.getCarrier());
        dto.setTrackingNo(shipping.getTrackingNo());
        dto.setShippingMethod(shipping.getShippingMethod());
        dto.setOriginAddress(shipping.getOriginAddress());
        dto.setDestinationAddress(shipping.getDestinationAddress());
        dto.setShippingDate(shipping.getShippingDate());
        dto.setEstimatedArrivalDate(shipping.getEstimatedArrivalDate());
        dto.setStatus(shipping.getStatus());
        dto.setCreatedAt(shipping.getCreatedAt());
        dto.setUpdatedAt(shipping.getUpdatedAt());
        return dto;
    }
}
