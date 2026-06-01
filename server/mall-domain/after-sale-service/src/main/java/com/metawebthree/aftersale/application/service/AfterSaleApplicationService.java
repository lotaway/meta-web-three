package com.metawebthree.aftersale.application.service;

import com.metawebthree.aftersale.application.dto.AfterSaleApplyDTO;
import com.metawebthree.aftersale.application.dto.AfterSaleDTO;
import com.metawebthree.aftersale.application.dto.AfterSaleProcessDTO;
import com.metawebthree.aftersale.domain.model.AfterSaleOrderDO;
import com.metawebthree.aftersale.domain.model.AfterSaleStatus;
import com.metawebthree.aftersale.domain.model.AfterSaleType;
import com.metawebthree.aftersale.domain.repository.AfterSaleRepository;
import com.metawebthree.aftersale.infrastructure.client.OrderClient;
import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.enums.ResponseStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class AfterSaleApplicationService {

    private final AfterSaleRepository afterSaleRepository;
    private final OrderClient orderClient;

    public AfterSaleApplicationService(AfterSaleRepository afterSaleRepository, OrderClient orderClient) {
        this.afterSaleRepository = afterSaleRepository;
        this.orderClient = orderClient;
    }

    /**
     * Apply for after-sale
     */
    @Transactional
    public AfterSaleDTO apply(AfterSaleApplyDTO applyDTO, Long userId) {
        // Verify order exists
        var orderDTO = orderClient.getOrderById(applyDTO.getOrderId());
        if (orderDTO == null) {
            throw new BusinessException(ResponseStatus.NOT_FOUND, "Order not found");
        }

        // Verify user owns the order
        if (orderDTO.getUserId() != userId) {
            throw new BusinessException(ResponseStatus.FORBIDDEN, "Not authorized to apply for this order");
        }

        // Create after-sale record
        AfterSaleOrderDO afterSaleOrder = new AfterSaleOrderDO();
        afterSaleOrder.setOrderId(applyDTO.getOrderId());
        afterSaleOrder.setOrderNo(orderDTO.getOrderNo());
        afterSaleOrder.setUserId(userId);
        afterSaleOrder.setProductId(applyDTO.getProductId());
        afterSaleOrder.setSkuId(applyDTO.getSkuId());
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