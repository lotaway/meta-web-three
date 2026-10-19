package com.metawebthree.payment.interfaces.web;

import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.payment.application.dto.ExchangeOrderRequest;
import com.metawebthree.payment.application.dto.ExchangeOrderResponse;
import com.metawebthree.payment.application.ExchangeOrderServiceImpl;
import com.metawebthree.payment.application.util.PaymentCallbackSignature;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import io.swagger.v3.oas.annotations.tags.Tags;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import jakarta.validation.Valid;
import java.util.List;

@RestController
@RequestMapping("/exchange")
@RequiredArgsConstructor
@Slf4j
@Tags(value = { @Tag(name = "Exchange"), @Tag(name = "Order"), @Tag(name = "Payment") })
public class ExchangeOrderController {

    private final ExchangeOrderServiceImpl exchangeOrderService;

    @Value("${payment.exchange.callback-secret:}")
    private String callbackSecret;

    @PostMapping("/orders")
    public ResponseEntity<ExchangeOrderResponse> createOrder(
            @Valid @RequestBody ExchangeOrderRequest request,
            @RequestHeader(HeaderConstants.USER_ID) Long userId) {

        log.info("Creating exchange order for user {}: {}", userId, request);

        ExchangeOrderResponse response = exchangeOrderService.createOrder(request, userId);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/orders/{orderNo}")
    public ResponseEntity<ExchangeOrderResponse> getOrder(
            @PathVariable String orderNo,
            @RequestHeader(HeaderConstants.USER_ID) Long userId) {

        log.info("Getting order details for user {}: {}", userId, orderNo);

        ExchangeOrderResponse response = exchangeOrderService.getOrder(orderNo, userId);

        return ResponseEntity.ok(response);
    }

    @GetMapping("/orders")
    @Operation(summary = "Get user order list")
    public ResponseEntity<List<ExchangeOrderResponse>> getUserOrders(
            @RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestParam(required = false) String status) {

        log.info("Getting orders for user {} with status: {}", userId, status);

        List<ExchangeOrderResponse> orders = exchangeOrderService.getUserOrders(userId, status);

        return ResponseEntity.ok(orders);
    }

    @DeleteMapping("/orders/{orderNo}")
    @Operation(summary = "Cancel user order")
    public ResponseEntity<Void> cancelOrder(
            @PathVariable String orderNo,
            @RequestHeader(HeaderConstants.USER_ID) Long userId) {

        log.info("Cancelling order for user {}: {}", userId, orderNo);

        exchangeOrderService.cancelOrder(orderNo, userId);

        return ResponseEntity.ok().build();
    }

    @PostMapping("/payment/callback")
    @Operation(summary = "Payment callback")
    public ResponseEntity<Void> paymentCallback(
            @RequestParam String paymentOrderNo,
            @RequestParam String status,
            @RequestParam String transactionId,
            @RequestHeader(value = "X-Payment-Signature", required = false) String signature) {

        log.info("Payment callback received: orderNo={}, status={}, txId={}",
                paymentOrderNo, status, transactionId);

        if (!PaymentCallbackSignature.verify(callbackSecret, paymentOrderNo, status, transactionId, signature)) {
            log.warn("Rejected exchange payment callback with invalid signature: orderNo={}", paymentOrderNo);
            return ResponseEntity.status(403).build();
        }

        exchangeOrderService.handlePaymentCallback(paymentOrderNo, status, transactionId);

        return ResponseEntity.ok().build();
    }
}
