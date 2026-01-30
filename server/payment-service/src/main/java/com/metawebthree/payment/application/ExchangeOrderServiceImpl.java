package com.metawebthree.payment.application;

import com.metawebthree.common.annotations.LogMethod;
import com.metawebthree.dto.ExchangeOrderRequest;
import com.metawebthree.dto.ExchangeOrderResponse;
import com.metawebthree.entity.ExchangeOrder;
import com.metawebthree.entity.UserKYC;
import com.metawebthree.repository.ExchangeOrderRepository;
import com.metawebthree.repository.UserKYCRepository;
import com.metawebthree.service.PaymentService;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;

/**
 * 兑换订单服务
 *
 * @TODO: 如需对接自定义KYC、风控、支付、汇率等服务，请在本类中注入自定义实现，
 * 并在 createOrder、validateUserKYC、processPayment、processCryptoTransfer 等方法中调用。
 * 推荐将第三方服务的接口抽象为独立Service，便于后续扩展和切换。
 *
 * 示例：
 * 1. 注入自定义KYCSeExchangeOrderServiceImpliskControlService实现
 * 3. 注入自定义PaymentService实现
 * 4. 注入自定义PriceEngineService实现
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class ExchangeOrderServiceImpl {

    private final ExchangeOrderRepository exchangeOrderRepository;
    private final UserKYCRepository userKYCRepository;
    private final PriceEngineServiceImpl priceEngineService;
    private final RiskControlServiceImpl riskControlService;
    private final PaymentService paymentService;
    private final CryptoWalletServiceImpl cryptoWalletService;

    @Value("${payment.risk-control.single-limit.usd:10000}")
    private BigDecimal singleLimitUSD;

    @Value("${payment.risk-control.daily-limit.usd:50000}")
    private BigDecimal dailyLimitUSD;

    @Transactional
    public ExchangeOrderResponse createOrder(ExchangeOrderRequest request, Long userId) {
        log.info("Creating exchange order for user {}: {}", userId, request);
        validateUserKYC(userId, request.getAmount(), request.getFiatCurrency());
        riskControlService.validateOrder(userId, request.getAmount(), request.getFiatCurrency());
        BigDecimal exchangeRate = getExchangeRate(request);
        BigDecimal cryptoAmount = calculateCryptoAmount(request.getAmount(), exchangeRate, request.getOrderType());
        ExchangeOrder order = buildOrder(request, userId, exchangeRate, cryptoAmount);
        exchangeOrderRepository.insert(order);
        ExchangeOrderResponse response = buildResponse(order);
        if (request.getAutoExecute()) {
            processPayment(order);
        }

        log.info("Created exchange order: {}", order.getOrderNo());
        return response;
    }

    public ExchangeOrderResponse getOrder(String orderNo, Long userId) {
        ExchangeOrder order = exchangeOrderRepository.findByOrderNo(orderNo);
        if (order == null) {
            throw new RuntimeException("Order not found: " + orderNo);
        }

        if (!order.getUserId().equals(userId)) {
            throw new RuntimeException("Unauthorized access to order");
        }

        return buildResponse(order);
    }

    public List<ExchangeOrderResponse> getUserOrders(Long userId, String status) {
        List<ExchangeOrder> orders;
        if (status != null && !status.isEmpty()) {
            orders = exchangeOrderRepository.findByUserIdAndStatus(userId, status);
        } else {
            orders = exchangeOrderRepository.findByUserId(userId);
        }

        return orders.stream()
                .map(this::buildResponse)
                .toList();
    }

    @LogMethod
    @Transactional
    public void cancelOrder(String orderNo, Long userId) {
        ExchangeOrder order = exchangeOrderRepository.findByOrderNo(orderNo);
        if (order == null) {
            throw new RuntimeException("Order not found: " + orderNo);
        }
        if (!order.getUserId().equals(userId)) {
            throw new RuntimeException("Unauthorized access to order");
        }
        if (order.getStatus() != ExchangeOrder.OrderStatus.PENDING) {
            throw new RuntimeException("Cannot cancel order with status: " + order.getStatus());
        }
        order.setStatus(ExchangeOrder.OrderStatus.CANCELLED);
        exchangeOrderRepository.updateById(order);
    }

    @Transactional
    public void handlePaymentCallback(String paymentOrderNo, String status, String transactionId) {
        ExchangeOrder order = exchangeOrderRepository.findByPaymentOrderNo(paymentOrderNo);
        if (order == null) {
            throw new RuntimeException("Order not found for payment: " + paymentOrderNo);
        }

        if ("SUCCESS".equals(status)) {
            order.setStatus(ExchangeOrder.OrderStatus.PAID);
            order.setPaidAt(LocalDateTime.now());
            exchangeOrderRepository.updateById(order);
            processCryptoTransfer(order);
        } else {
            order.setStatus(ExchangeOrder.OrderStatus.FAILED);
            order.setFailureReason("Payment failed: " + status);
            exchangeOrderRepository.updateById(order);
        }

        log.info("Payment callback processed for order {}: {}", order.getOrderNo(), status);
    }

    private void validateUserKYC(Long userId, BigDecimal amount, String fiatCurrency) {
        UserKYC kyc = userKYCRepository.findHighestApprovedLevelByUserId(userId);

        if (kyc == null) {
            throw new RuntimeException("KYC verification required");
        }

        BigDecimal limit = BigDecimal.valueOf(kyc.getLevel().getLimit());
        BigDecimal usdAmount = convertToUSD(amount, fiatCurrency);

        if (usdAmount.compareTo(limit) > 0) {
            throw new RuntimeException("Amount exceeds KYC level limit. Current level: " +
                    kyc.getLevel().getDescription() + ", Limit: " + limit + " USD");
        }
    }

    private BigDecimal getExchangeRate(ExchangeOrderRequest request) {
        if ("BUY_CRYPTO".equals(request.getOrderType())) {
            return priceEngineService.getWeightedAveragePrice(request.getCryptoCurrency(), request.getFiatCurrency());
        } else {
            return priceEngineService.getWeightedAveragePrice(request.getFiatCurrency(), request.getCryptoCurrency());
        }
    }

    private BigDecimal calculateCryptoAmount(BigDecimal fiatAmount, BigDecimal exchangeRate, String orderType) {
        if ("BUY_CRYPTO".equals(orderType)) {
            return fiatAmount.divide(exchangeRate, 8, RoundingMode.HALF_UP);
        } else {
            return fiatAmount.multiply(exchangeRate).setScale(8, RoundingMode.HALF_UP);
        }
    }

    private ExchangeOrder buildOrder(ExchangeOrderRequest request, Long userId, BigDecimal exchangeRate,
            BigDecimal cryptoAmount) {
        String orderNo = generateOrderNo();

        return ExchangeOrder.builder()
                .orderNo(orderNo)
                .userId(userId)
                .orderType(ExchangeOrder.OrderType.valueOf(request.getOrderType()))
                .status(ExchangeOrder.OrderStatus.PENDING)
                .fiatCurrency(request.getFiatCurrency())
                .cryptoCurrency(request.getCryptoCurrency())
                .fiatAmount(request.getAmount())
                .cryptoAmount(cryptoAmount)
                .exchangeRate(exchangeRate)
                .paymentMethod(ExchangeOrder.PaymentMethod.valueOf(request.getPaymentMethod()))
                .userWalletAddress(request.getWalletAddress())
                .kycLevel(request.getKycLevel())
                .kycVerified(true)
                .expiredAt(LocalDateTime.now().plusMinutes(30))
                .remark(request.getRemark())
                .build();
    }

    private ExchangeOrderResponse buildResponse(ExchangeOrder order) {
        return ExchangeOrderResponse.builder()
                .orderNo(order.getOrderNo())
                .status(order.getStatus().name())
                .orderType(order.getOrderType().name())
                .fiatCurrency(order.getFiatCurrency())
                .cryptoCurrency(order.getCryptoCurrency())
                .fiatAmount(order.getFiatAmount())
                .cryptoAmount(order.getCryptoAmount())
                .exchangeRate(order.getExchangeRate())
                .paymentMethod(order.getPaymentMethod().name())
                .walletAddress(order.getUserWalletAddress())
                .createdAt(order.getCreatedAt())
                .expiredAt(order.getExpiredAt())
                .kycLevel(order.getKycLevel())
                .kycVerified(order.getKycVerified())
                .remark(order.getRemark())
                .build();
    }

    private void processPayment(ExchangeOrder order) {
        try {
            String paymentUrl = paymentService.createPayment(order);
            // @TODO: Update paymentUrl in order
            log.info("Payment URL generated for order {}: {}", order.getOrderNo(), paymentUrl);
        } catch (Exception e) {
            log.error("Failed to create payment for order {}: {}", order.getOrderNo(), e.getMessage());
            order.setStatus(ExchangeOrder.OrderStatus.FAILED);
            order.setFailureReason("Payment creation failed: " + e.getMessage());
            exchangeOrderRepository.updateById(order);
        }
    }

    private void processCryptoTransfer(ExchangeOrder order) {
        try {
            String txHash = cryptoWalletService.transferCrypto(order);
            order.setCryptoTransactionHash(txHash);
            order.setStatus(ExchangeOrder.OrderStatus.COMPLETED);
            order.setCompletedAt(LocalDateTime.now());
            exchangeOrderRepository.updateById(order);

            log.info("Crypto transfer completed for order {}: {}", order.getOrderNo(), txHash);
        } catch (Exception e) {
            log.error("Failed to transfer crypto for order {}: {}", order.getOrderNo(), e.getMessage());
            order.setStatus(ExchangeOrder.OrderStatus.FAILED);
            order.setFailureReason("Crypto transfer failed: " + e.getMessage());
            exchangeOrderRepository.updateById(order);
        }
    }

    private String generateOrderNo() {
        return "EX" + System.currentTimeMillis() + UUID.randomUUID().toString().substring(0, 8).toUpperCase();
    }

    // @TODO: Use external real price api
    private BigDecimal convertToUSD(BigDecimal amount, String currency) {
        return switch (currency) {
            case "USD" -> amount;
            case "CNY" -> amount.divide(new BigDecimal("7.0"), 2, RoundingMode.HALF_UP);
            case "EUR" -> amount.multiply(new BigDecimal("1.1"));
            default -> amount;
        };
    }
}