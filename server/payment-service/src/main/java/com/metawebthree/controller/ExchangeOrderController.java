package com.metawebthree.controller;

import com.metawebthree.dto.ExchangeOrderRequest;
import com.metawebthree.dto.ExchangeOrderResponse;
import com.metawebthree.service.ExchangeOrderService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import jakarta.validation.Valid;
import java.util.List;

@RestController
@RequestMapping("/api/v1/exchange")
@RequiredArgsConstructor
@Slf4j
public class ExchangeOrderController {
    
    private final ExchangeOrderService exchangeOrderService;
    
    /**
     * 创建兑换订单
     */
    @PostMapping("/orders")
    public ResponseEntity<ExchangeOrderResponse> createOrder(
            @Valid @RequestBody ExchangeOrderRequest request,
            @RequestHeader("X-User-ID") Long userId) {
        
        log.info("Creating exchange order for user {}: {}", userId, request);
        
        ExchangeOrderResponse response = exchangeOrderService.createOrder(request, userId);
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * 获取订单详情
     */
    @GetMapping("/orders/{orderNo}")
    public ResponseEntity<ExchangeOrderResponse> getOrder(
            @PathVariable String orderNo,
            @RequestHeader("X-User-ID") Long userId) {
        
        log.info("Getting order details for user {}: {}", userId, orderNo);
        
        ExchangeOrderResponse response = exchangeOrderService.getOrder(orderNo, userId);
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * 获取用户订单列表
     */
    @GetMapping("/orders")
    public ResponseEntity<List<ExchangeOrderResponse>> getUserOrders(
            @RequestHeader("X-User-ID") Long userId,
            @RequestParam(required = false) String status) {
        
        log.info("Getting orders for user {} with status: {}", userId, status);
        
        List<ExchangeOrderResponse> orders = exchangeOrderService.getUserOrders(userId, status);
        
        return ResponseEntity.ok(orders);
    }
    
    /**
     * 取消订单
     */
    @DeleteMapping("/orders/{orderNo}")
    public ResponseEntity<Void> cancelOrder(
            @PathVariable String orderNo,
            @RequestHeader("X-User-ID") Long userId) {
        
        log.info("Cancelling order for user {}: {}", userId, orderNo);
        
        exchangeOrderService.cancelOrder(orderNo, userId);
        
        return ResponseEntity.ok().build();
    }
    
    /**
     * 支付回调
     */
    @PostMapping("/payment/callback")
    public ResponseEntity<Void> paymentCallback(
            @RequestParam String paymentOrderNo,
            @RequestParam String status,
            @RequestParam(required = false) String transactionId) {
        
        log.info("Payment callback received: orderNo={}, status={}, txId={}", 
                paymentOrderNo, status, transactionId);
        
        exchangeOrderService.handlePaymentCallback(paymentOrderNo, status, transactionId);
        
        return ResponseEntity.ok().build();
    }
} 