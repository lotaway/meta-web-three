package com.metawebthree.order.application.saga;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.order.application.OrderApplicationService;
import com.metawebthree.order.domain.model.OrderDO;
import com.metawebthree.order.domain.model.SagaInstance;
import com.metawebthree.order.domain.model.SagaStep;
import com.metawebthree.order.infrastructure.client.InventoryClient;
import com.metawebthree.order.infrastructure.client.PromotionClient;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderMapper;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

/**
 * Order payment saga orchestrator.
 * Handles the distributed transaction: Order Create -> Inventory Reserve -> Payment -> Inventory Confirm
 */
@Slf4j
@Component
public class OrderPaymentSaga {

    @Autowired
    private SagaOrchestrator sagaOrchestrator;

    @Autowired
    private OrderApplicationService orderApplicationService;

    @Autowired
    private OrderMapper orderMapper;

    @Autowired
    private InventoryClient inventoryClient;

    @Autowired
    private PromotionClient promotionClient;

    @Autowired
    private ObjectMapper objectMapper;

    /**
     * Execute the order payment saga.
     * 
     * @param userId user ID
     * @param remark order remark
     * @param items order items
     * @param memberReceiveAddressId address ID
     * @param couponId coupon ID
     * @param useIntegration integration points to use
     * @return order ID if success, null if failed
     */
    public Long execute(Long userId, String remark, List<OrderApplicationService.OrderItemCreate> items,
                        Long memberReceiveAddressId, Long couponId, Integer useIntegration) {
        
        String bizId = "ORDER_" + System.currentTimeMillis();
        
        // Define saga steps
        List<SagaOrchestrator.SagaStepDefinition> steps = new ArrayList<>();
        steps.add(new SagaOrchestrator.SagaStepDefinition(
            SagaStep.StepName.CREATE_ORDER, "order-service", 1, true));
        steps.add(new SagaOrchestrator.SagaStepDefinition(
            SagaStep.StepName.RESERVE_INVENTORY, "inventory-service", 2, true));
        steps.add(new SagaOrchestrator.SagaStepDefinition(
            SagaStep.StepName.PROCESS_PAYMENT, "payment-service", 3, true));
        steps.add(new SagaOrchestrator.SagaStepDefinition(
            SagaStep.StepName.CONFIRM_INVENTORY, "inventory-service", 4, false));
        
        // Start saga
        SagaInstance sagaInstance = sagaOrchestrator.startSaga(
            SagaInstance.Type.ORDER_PAYMENT_SAGA, bizId, steps);
        
        String sagaId = sagaInstance.getSagaId();
        Long orderId = null;
        
        try {
            // Step 1: Create Order
            log.info("Saga step 1: Creating order - sagaId={}", sagaId);
            orderId = orderApplicationService.createOrder(userId, remark, items, 
                memberReceiveAddressId, couponId, useIntegration);
            
            OrderCreatedData orderData = new OrderCreatedData();
            orderData.setOrderId(orderId);
            orderData.setUserId(userId);
            orderData.setTotalAmount(items.stream()
                .map(i -> i.getUnitPrice().multiply(BigDecimal.valueOf(i.getQuantity())))
                .reduce(BigDecimal.ZERO, BigDecimal::add));
            
            SagaOrchestrator.SagaStepResult step1Result = SagaOrchestrator.SagaStepResult.success(
                orderData, new OrderCompensationData(orderId, userId));
            
            sagaOrchestrator.executeStep(sagaId, SagaStep.StepName.CREATE_ORDER, orderData, 
                req -> step1Result, req -> null);
            
            // Step 2: Reserve Inventory
            log.info("Saga step 2: Reserving inventory - sagaId={}", sagaId);
            InventoryReserveRequest reserveRequest = new InventoryReserveRequest();
            reserveRequest.setOrderId(orderId);
            reserveRequest.setItems(items);
            final Long reserveOrderId = orderId;
            
            boolean reserveResult = inventoryClient.reserveInventory(
                orderId, items.stream().map(i -> i.getSkuId()).toList(),
                items.stream().map(i -> i.getQuantity()).toList(),
                "Order payment saga - reserve inventory");
            
            if (!reserveResult) {
                throw new RuntimeException("Failed to reserve inventory");
            }
            
            SagaOrchestrator.SagaStepResult step2Result = SagaOrchestrator.SagaStepResult.success(
                reserveRequest, new InventoryCompensationData(orderId));
            final Long step2OrderId = orderId;
            
            sagaOrchestrator.executeStep(sagaId, SagaStep.StepName.RESERVE_INVENTORY, reserveRequest,
                req -> step2Result, req -> {
                    inventoryClient.releaseInventoryByOrderId(step2OrderId, "Saga compensation - release inventory");
                    return null;
                });
            
            // Step 3: Process Payment
            log.info("Saga step 3: Processing payment - sagaId={}", sagaId);
            PaymentRequest paymentRequest = new PaymentRequest();
            paymentRequest.setOrderId(orderId);
            paymentRequest.setUserId(userId);
            paymentRequest.setAmount(orderData.getTotalAmount());
            
            // Call payment service (this would be a real RPC call in production)
            boolean paymentResult = true; // paymentService.processPayment(userId, orderId, orderData.getTotalAmount());
            
            if (!paymentResult) {
                throw new RuntimeException("Payment failed");
            }
            
            SagaOrchestrator.SagaStepResult step3Result = SagaOrchestrator.SagaStepResult.success(
                paymentRequest, new PaymentCompensationData(orderId));
            final Long step3OrderId = orderId;
            final BigDecimal step3Amount = orderData.getTotalAmount();
            
            sagaOrchestrator.executeStep(sagaId, SagaStep.StepName.PROCESS_PAYMENT, paymentRequest,
                req -> step3Result, req -> {
                    // Call payment refund
                    // paymentService.refund(orderId);
                    log.info("Saga compensation: refund for order {}, amount: {}", step3OrderId, step3Amount);
                    return null;
                });
            
            // Step 4: Confirm Inventory (deduct from reserved)
            log.info("Saga step 4: Confirming inventory - sagaId={}", sagaId);
            boolean confirmResult = inventoryClient.confirmInventoryReservation(orderId, 
                "Order payment saga - confirm inventory");
            
            if (!confirmResult) {
                throw new RuntimeException("Failed to confirm inventory");
            }
            
            SagaOrchestrator.SagaStepResult step4Result = SagaOrchestrator.SagaStepResult.success(
                new InventoryConfirmRequest(orderId), null);
            
            sagaOrchestrator.executeStep(sagaId, SagaStep.StepName.CONFIRM_INVENTORY, 
                new InventoryConfirmRequest(orderId), req -> step4Result, req -> null);
            
            // Update order status to PAID
            orderApplicationService.paySuccess(orderId, 1);
            
            // Mark saga as completed
            sagaOrchestrator.updateSagaStatus(sagaId, SagaInstance.Status.COMPLETED, null);
            
            log.info("Order payment saga completed successfully: sagaId={}, orderId={}", sagaId, orderId);
            return orderId;
            
        } catch (Exception e) {
            log.error("Order payment saga failed: sagaId={}, error={}", sagaId, e.getMessage(), e);
            
            // Compensate all completed steps
            sagaOrchestrator.compensate(sagaId);
            
            // Update order status to FAILED
            if (orderId != null) {
                OrderDO order = orderMapper.selectById(orderId);
                if (order != null) {
                    order.setOrderStatus("FAILED");
                    orderMapper.updateById(order);
                }
            }
            
            throw new RuntimeException("Order payment saga failed: " + e.getMessage(), e);
        }
    }

    // Request/Response DTOs
    @Data
    public static class OrderCreatedData {
        private Long orderId;
        private Long userId;
        private BigDecimal totalAmount;
    }

    @Data
    public static class OrderCompensationData {
        private Long orderId;
        private Long userId;
        
        public OrderCompensationData() {}
        public OrderCompensationData(Long orderId, Long userId) {
            this.orderId = orderId;
            this.userId = userId;
        }
    }

    @Data
    public static class InventoryReserveRequest {
        private Long orderId;
        private List<OrderApplicationService.OrderItemCreate> items;
    }

    @Data
    public static class InventoryCompensationData {
        private Long orderId;
        
        public InventoryCompensationData() {}
        public InventoryCompensationData(Long orderId) {
            this.orderId = orderId;
        }
    }

    @Data
    public static class PaymentRequest {
        private Long orderId;
        private Long userId;
        private BigDecimal amount;
    }

    @Data
    public static class PaymentCompensationData {
        private Long orderId;
        
        public PaymentCompensationData() {}
        public PaymentCompensationData(Long orderId) {
            this.orderId = orderId;
        }
    }

    @Data
    public static class InventoryConfirmRequest {
        private Long orderId;
        
        public InventoryConfirmRequest() {}
        public InventoryConfirmRequest(Long orderId) {
            this.orderId = orderId;
        }
    }
}
