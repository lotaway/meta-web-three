package com.metawebthree.aftersale.application.service;

import com.metawebthree.aftersale.application.dto.AfterSaleApplyDTO;
import com.metawebthree.aftersale.application.dto.AfterSaleDTO;
import com.metawebthree.aftersale.application.dto.AfterSaleProcessDTO;
import com.metawebthree.aftersale.application.dto.AfterSaleQueryDTO;
import com.metawebthree.aftersale.application.dto.AfterSaleStatisticDTO;
import com.metawebthree.aftersale.domain.model.AfterSaleOrderDO;
import com.metawebthree.aftersale.domain.model.AfterSaleStatus;
import com.metawebthree.aftersale.domain.model.AfterSaleType;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.aftersale.domain.repository.AfterSaleRepository;
import com.metawebthree.aftersale.infrastructure.client.OrderClient;
import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.order.domain.model.OrderDO;
import com.metawebthree.order.domain.model.OrderItemDO;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderItemMapper;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderMapper;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

@Service
public class AfterSaleApplicationService {

    private final AfterSaleRepository afterSaleRepository;
    private final OrderClient orderClient;
    private final OrderMapper orderMapper;
    private final OrderItemMapper orderItemMapper;

    public AfterSaleApplicationService(AfterSaleRepository afterSaleRepository, OrderClient orderClient,
            OrderMapper orderMapper, OrderItemMapper orderItemMapper) {
        this.afterSaleRepository = afterSaleRepository;
        this.orderClient = orderClient;
        this.orderMapper = orderMapper;
        this.orderItemMapper = orderItemMapper;
    }

    /**
     * Apply for after-sale
     */
    @Transactional
    public AfterSaleDTO apply(AfterSaleApplyDTO applyDTO, Long userId) {
        if (applyDTO == null || applyDTO.getOrderId() == null) {
            throw new BusinessException(ResponseStatus.PARAM_MISSING_ERROR, "orderId is required");
        }
        if (applyDTO.getSkuId() == null) {
            throw new BusinessException(ResponseStatus.PARAM_MISSING_ERROR, "skuId is required");
        }
        if (applyDTO.getQuantity() == null || applyDTO.getQuantity() <= 0) {
            throw new BusinessException(ResponseStatus.PARAM_ERROR, "After-sale quantity must be positive");
        }
        if (applyDTO.getRefundAmount() == null || applyDTO.getRefundAmount() <= 0) {
            throw new BusinessException(ResponseStatus.PARAM_ERROR, "Refund amount must be positive");
        }
        if (!isValidAfterSaleType(applyDTO.getAfterSaleType())) {
            throw new BusinessException(ResponseStatus.PARAM_ERROR, "Invalid after-sale type");
        }

        // Verify order exists and belongs to the user
        OrderDO order = orderMapper.selectById(applyDTO.getOrderId());
        if (order == null) {
            throw new BusinessException(ResponseStatus.ORDER_NOT_FOUND, "Order not found");
        }
        if (!Objects.equals(order.getUserId(), userId)) {
            throw new BusinessException(ResponseStatus.FORBIDDEN, "Not authorized to apply for this order");
        }
        if (!isAfterSaleEligibleStatus(order.getOrderStatus())) {
            throw new BusinessException(ResponseStatus.ORDER_STATUS_INVALID,
                    "Order status does not allow after-sale application");
        }

        // The requested SKU must belong to the order.
        OrderItemDO orderItem = resolveOrderItem(applyDTO);
        if (orderItem == null) {
            throw new BusinessException(ResponseStatus.PARAM_ERROR,
                    "The requested product/SKU does not belong to this order");
        }

        int orderedQuantity = orderItem.getQuantity() == null ? 0 : orderItem.getQuantity();
        int alreadyApplied = sumActiveAfterSaleQuantity(orderItem.getOrderId(), orderItem.getSkuId());
        if (applyDTO.getQuantity() + alreadyApplied > orderedQuantity) {
            throw new BusinessException(ResponseStatus.PARAM_ERROR,
                    "After-sale quantity exceeds the remaining refundable quantity for this item");
        }

        int lineTotalCents = toCents(orderItem.getUnitPrice()) * orderedQuantity;
        if (applyDTO.getRefundAmount() > lineTotalCents) {
            throw new BusinessException(ResponseStatus.PARAM_ERROR,
                    "Refund amount exceeds the amount paid for this item");
        }

        AfterSaleOrderDO afterSaleOrder = new AfterSaleOrderDO();
        afterSaleOrder.setOrderId(order.getId());
        afterSaleOrder.setOrderNo(order.getOrderNo());
        afterSaleOrder.setUserId(userId);
        afterSaleOrder.setProductId(orderItem.getProductId());
        afterSaleOrder.setSkuId(orderItem.getSkuId());
        afterSaleOrder.setProductName(orderItem.getProductName());
        afterSaleOrder.setProductImage(orderItem.getImageUrl());
        afterSaleOrder.setQuantity(applyDTO.getQuantity());
        afterSaleOrder.setRefundAmount(applyDTO.getRefundAmount());
        afterSaleOrder.setAfterSaleType(applyDTO.getAfterSaleType());
        afterSaleOrder.setAfterSaleStatus(AfterSaleStatus.PENDING.getCode());
        afterSaleOrder.setApplyReason(applyDTO.getApplyReason());
        afterSaleOrder.setApplyTime(LocalDateTime.now());
        afterSaleOrder.setRemark(applyDTO.getRemark());

        afterSaleRepository.save(afterSaleOrder);

        // Notify order service to create return apply
        orderClient.createReturnApply(applyDTO.getOrderId(), applyDTO.getApplyReason(), userId);

        return convertToDTO(afterSaleOrder);
    }

    private OrderItemDO resolveOrderItem(AfterSaleApplyDTO applyDTO) {
        List<OrderItemDO> items = orderItemMapper.selectList(new LambdaQueryWrapper<OrderItemDO>()
                .eq(OrderItemDO::getOrderId, applyDTO.getOrderId())
                .eq(OrderItemDO::getSkuId, applyDTO.getSkuId()));
        if (items.isEmpty()) {
            return null;
        }
        if (applyDTO.getProductId() != null) {
            return items.stream()
                    .filter(item -> Objects.equals(item.getProductId(), applyDTO.getProductId()))
                    .findFirst()
                    .orElse(null);
        }
        return items.get(0);
    }

    private int sumActiveAfterSaleQuantity(Long orderId, Long skuId) {
        if (orderId == null || skuId == null) {
            return 0;
        }
        List<AfterSaleOrderDO> existing = afterSaleRepository.findByOrderId(orderId);
        int sum = 0;
        for (AfterSaleOrderDO record : existing) {
            if (!Objects.equals(record.getSkuId(), skuId)) {
                continue;
            }
            Integer status = record.getAfterSaleStatus();
            if (status == null
                    || status.equals(AfterSaleStatus.REJECTED.getCode())
                    || status.equals(AfterSaleStatus.CANCELLED.getCode())) {
                continue;
            }
            sum += record.getQuantity() == null ? 0 : record.getQuantity();
        }
        return sum;
    }

    private boolean isValidAfterSaleType(Integer type) {
        if (type == null) {
            return false;
        }
        for (AfterSaleType candidate : AfterSaleType.values()) {
            if (candidate.getCode().equals(type)) {
                return true;
            }
        }
        return false;
    }

    private boolean isAfterSaleEligibleStatus(String status) {
        return "PAID".equals(status) || "SHIPPED".equals(status) || "COMPLETED".equals(status);
    }

    private int toCents(BigDecimal amount) {
        if (amount == null) {
            return 0;
        }
        return amount.movePointRight(2).intValue();
    }

    /**
     * Process after-sale application (approve/reject)
     */
    @Transactional
    public AfterSaleDTO process(AfterSaleProcessDTO processDTO) {
        AfterSaleOrderDO afterSaleOrder = afterSaleRepository.findById(processDTO.getId());
        if (afterSaleOrder == null) {
            throw new BusinessException(ResponseStatus.NOT_FOUND, "After-sale record not found");
        }

        if (processDTO.getStatus().equals(AfterSaleStatus.APPROVED.getCode())) {
            // Close the order for refund
            orderClient.closeOrderForRefund(
                String.valueOf(afterSaleOrder.getOrderId()),
                "After-sale approved: " + (processDTO.getRemark() != null ? processDTO.getRemark() : "")
            );
            afterSaleOrder.setAfterSaleStatus(AfterSaleStatus.APPROVED.getCode());
        } else if (processDTO.getStatus().equals(AfterSaleStatus.REJECTED.getCode())) {
            afterSaleOrder.setAfterSaleStatus(AfterSaleStatus.REJECTED.getCode());
            afterSaleOrder.setRejectReason(processDTO.getRejectReason());
        } else {
            throw new BusinessException(ResponseStatus.PARAM_ERROR, "Invalid after-sale type");
        }

        afterSaleOrder.setProcessTime(LocalDateTime.now());
        if (processDTO.getRemark() != null) {
            afterSaleOrder.setRemark(processDTO.getRemark());
        }

        afterSaleRepository.save(afterSaleOrder);
        return convertToDTO(afterSaleOrder);
    }

    /**
     * Get after-sale by ID
     */
    public AfterSaleDTO getById(Long id) {
        AfterSaleOrderDO afterSaleOrder = afterSaleRepository.findById(id);
        if (afterSaleOrder == null) {
            throw new BusinessException(ResponseStatus.NOT_FOUND, "After-sale record not found");
        }
        return convertToDTO(afterSaleOrder);
    }

    /**
     * Get after-sale list by user ID
     */
    public List<AfterSaleDTO> getByUserId(Long userId) {
        return afterSaleRepository.findByUserId(userId).stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    /**
     * Get after-sale list by order ID
     */
    public List<AfterSaleDTO> getByOrderId(Long orderId) {
        return afterSaleRepository.findByOrderId(orderId).stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    /**
     * Get all after-sale records (admin)
     */
    public List<AfterSaleDTO> getAll() {
        return afterSaleRepository.findAll().stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
    }

    /**
     * Get all after-sale records with pagination (admin)
     */
    public Map<String, Object> getAllPaged(AfterSaleQueryDTO queryDTO) {
        List<AfterSaleOrderDO> list = afterSaleRepository.findByPage(queryDTO);
        Long total = afterSaleRepository.countByPage(queryDTO);
        
        List<AfterSaleDTO> dtoList = list.stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());
        
        Map<String, Object> result = new HashMap<>();
        result.put("data", dtoList);
        result.put("total", total);
        result.put("pageNum", queryDTO.getPageNum());
        result.put("pageSize", queryDTO.getPageSize());
        
        return result;
    }

    /**
     * Get after-sale statistics (admin)
     */
    public AfterSaleStatisticDTO getStatistics() {
        AfterSaleStatisticDTO stat = new AfterSaleStatisticDTO();
        
        stat.setTotalCount(afterSaleRepository.countTotal());
        stat.setPendingCount(afterSaleRepository.countByStatus(AfterSaleStatus.PENDING.getCode()));
        stat.setProcessingCount(afterSaleRepository.countByStatus(AfterSaleStatus.PROCESSING.getCode()));
        stat.setApprovedCount(afterSaleRepository.countByStatus(AfterSaleStatus.APPROVED.getCode()));
        stat.setRejectedCount(afterSaleRepository.countByStatus(AfterSaleStatus.REJECTED.getCode()));
        stat.setCompletedCount(afterSaleRepository.countByStatus(AfterSaleStatus.COMPLETED.getCode()));
        stat.setTotalRefundAmount(afterSaleRepository.sumRefundAmount());
        
        LocalDate today = LocalDate.now();
        LocalDate weekAgo = today.minusDays(7);
        LocalDate monthAgo = today.minusDays(30);
        
        String todayStr = today.format(DateTimeFormatter.ISO_DATE);
        String weekAgoStr = weekAgo.format(DateTimeFormatter.ISO_DATE);
        String monthAgoStr = monthAgo.format(DateTimeFormatter.ISO_DATE);
        
        stat.setTodayCount(afterSaleRepository.countByDateRange(todayStr, todayStr));
        stat.setWeekCount(afterSaleRepository.countByDateRange(weekAgoStr, todayStr));
        stat.setMonthCount(afterSaleRepository.countByDateRange(monthAgoStr, todayStr));
        
        return stat;
    }

    /**
     * Batch approve after-sale records
     */
    @Transactional
    public int batchApprove(List<Long> ids) {
        int count = 0;
        for (Long id : ids) {
            AfterSaleOrderDO afterSaleOrder = afterSaleRepository.findById(id);
            if (afterSaleOrder != null && afterSaleOrder.getAfterSaleStatus().equals(AfterSaleStatus.PENDING.getCode())) {
                orderClient.closeOrderForRefund(
                    String.valueOf(afterSaleOrder.getOrderId()),
                    "After-sale batch approved"
                );
                afterSaleRepository.updateStatus(id, AfterSaleStatus.APPROVED.getCode());
                count++;
            }
        }
        return count;
    }

    /**
     * Batch reject after-sale records
     */
    @Transactional
    public int batchReject(List<Long> ids, String reason) {
        int count = 0;
        for (Long id : ids) {
            AfterSaleOrderDO afterSaleOrder = afterSaleRepository.findById(id);
            if (afterSaleOrder != null && afterSaleOrder.getAfterSaleStatus().equals(AfterSaleStatus.PENDING.getCode())) {
                afterSaleOrder.setRejectReason(reason);
                afterSaleRepository.save(afterSaleOrder);
                afterSaleRepository.updateStatus(id, AfterSaleStatus.REJECTED.getCode());
                count++;
            }
        }
        return count;
    }

    private AfterSaleDTO convertToDTO(AfterSaleOrderDO afterSaleOrder) {
        AfterSaleDTO dto = new AfterSaleDTO();
        dto.setId(afterSaleOrder.getId());
        dto.setOrderId(afterSaleOrder.getOrderId());
        dto.setOrderNo(afterSaleOrder.getOrderNo());
        dto.setUserId(afterSaleOrder.getUserId());
        dto.setProductId(afterSaleOrder.getProductId());
        dto.setSkuId(afterSaleOrder.getSkuId());
        dto.setProductName(afterSaleOrder.getProductName());
        dto.setProductImage(afterSaleOrder.getProductImage());
        dto.setQuantity(afterSaleOrder.getQuantity());
        dto.setRefundAmount(afterSaleOrder.getRefundAmount());
        dto.setAfterSaleType(afterSaleOrder.getAfterSaleType());
        dto.setAfterSaleStatus(afterSaleOrder.getAfterSaleStatus());
        dto.setApplyReason(afterSaleOrder.getApplyReason());
        dto.setRejectReason(afterSaleOrder.getRejectReason());
        dto.setApplyTime(afterSaleOrder.getApplyTime());
        dto.setProcessTime(afterSaleOrder.getProcessTime());
        dto.setCompleteTime(afterSaleOrder.getCompleteTime());
        dto.setRemark(afterSaleOrder.getRemark());

        // Set type description
        if (afterSaleOrder.getAfterSaleType() != null) {
            for (AfterSaleType type : AfterSaleType.values()) {
                if (type.getCode().equals(afterSaleOrder.getAfterSaleType())) {
                    dto.setAfterSaleTypeDesc(type.getDesc());
                    break;
                }
            }
        }

        // Set status description
        if (afterSaleOrder.getAfterSaleStatus() != null) {
            for (AfterSaleStatus status : AfterSaleStatus.values()) {
                if (status.getCode().equals(afterSaleOrder.getAfterSaleStatus())) {
                    dto.setAfterSaleStatusDesc(status.getDesc());
                    break;
                }
            }
        }

        return dto;
    }
}