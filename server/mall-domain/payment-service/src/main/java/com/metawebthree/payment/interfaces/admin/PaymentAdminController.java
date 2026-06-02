package com.metawebthree.payment.interfaces.admin;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.payment.domain.model.UserKYC;
import com.metawebthree.payment.domain.model.CryptoPrice;
import com.metawebthree.payment.domain.model.ExchangeOrder;
import com.metawebthree.payment.infrastructure.persistence.mapper.UserKYCRepository;
import com.metawebthree.payment.infrastructure.persistence.mapper.CryptoPriceRepository;
import com.metawebthree.payment.infrastructure.persistence.mapper.ExchangeOrderRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/admin/payment")
public class PaymentAdminController {

    @Autowired
    private UserKYCRepository userKYCRepository;

    @Autowired
    private CryptoPriceRepository cryptoPriceRepository;

    @Autowired
    private ExchangeOrderRepository exchangeOrderRepository;

    @GetMapping("/kyc/list")
    public ApiResponse<Page<UserKYC>> listKYC(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) Long userId,
            @RequestParam(required = false) String level,
            @RequestParam(required = false) String status) {

        LambdaQueryWrapper<UserKYC> wrapper = new LambdaQueryWrapper<UserKYC>()
            .eq(userId != null, UserKYC::getUserId, userId)
            .eq(level != null, UserKYC::getLevel, level)
            .eq(status != null, UserKYC::getStatus, status)
            .orderByDesc(UserKYC::getCreatedAt);

        Page<UserKYC> page = new Page<>(pageNum, pageSize);
        Page<UserKYC> result = userKYCRepository.selectPage(page, wrapper);

        return ApiResponse.success(result);
    }

    @GetMapping("/kyc/{id}")
    public ApiResponse<UserKYC> getKYCById(@PathVariable Long id) {
        UserKYC kyc = userKYCRepository.selectById(id);
        if (kyc != null) {
            return ApiResponse.success(kyc);
        }
        return ApiResponse.error(ResponseStatus.NOT_FOUND, "KYC record not found");
    }

    @PutMapping("/kyc/{id}/review")
    public ApiResponse<Void> reviewKYC(
            @PathVariable Long id,
            @RequestBody Map<String, String> request) {
        UserKYC kyc = userKYCRepository.selectById(id);
        if (kyc == null) {
            return ApiResponse.error(ResponseStatus.NOT_FOUND, "KYC record not found");
        }

        String status = request.get("status");
        String reviewerId = request.get("reviewerId");
        String reviewNotes = request.get("reviewNotes");

        if (status != null) {
            kyc.setStatus(UserKYC.KYCStatus.valueOf(status));
        }
        if (reviewerId != null) {
            kyc.setReviewerId(reviewerId);
        }
        if (reviewNotes != null) {
            kyc.setReviewNotes(reviewNotes);
        }
        kyc.setReviewedAt(LocalDateTime.now());

        userKYCRepository.updateById(kyc);
        return ApiResponse.success();
    }

    @GetMapping("/kyc/statistics")
    public ApiResponse<Map<String, Object>> getKYCStatistics() {
        Long total = userKYCRepository.selectCount(null);

        LambdaQueryWrapper<UserKYC> pendingWrapper = new LambdaQueryWrapper<UserKYC>()
            .eq(UserKYC::getStatus, UserKYC.KYCStatus.PENDING);
        Long pending = userKYCRepository.selectCount(pendingWrapper);

        LambdaQueryWrapper<UserKYC> approvedWrapper = new LambdaQueryWrapper<UserKYC>()
            .eq(UserKYC::getStatus, UserKYC.KYCStatus.APPROVED);
        Long approved = userKYCRepository.selectCount(approvedWrapper);

        LambdaQueryWrapper<UserKYC> rejectedWrapper = new LambdaQueryWrapper<UserKYC>()
            .eq(UserKYC::getStatus, UserKYC.KYCStatus.REJECTED);
        Long rejected = userKYCRepository.selectCount(rejectedWrapper);

        Map<String, Object> data = new HashMap<>();
        data.put("total", total);
        data.put("pending", pending);
        data.put("approved", approved);
        data.put("rejected", rejected);
        return ApiResponse.success(data);
    }

    @GetMapping("/crypto-price/list")
    public ApiResponse<Page<CryptoPrice>> listCryptoPrices(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String symbol,
            @RequestParam(required = false) String baseCurrency,
            @RequestParam(required = false) String quoteCurrency,
            @RequestParam(required = false) String source) {

        LambdaQueryWrapper<CryptoPrice> wrapper = new LambdaQueryWrapper<CryptoPrice>()
            .like(symbol != null, CryptoPrice::getSymbol, symbol)
            .eq(baseCurrency != null, CryptoPrice::getBaseCurrency, baseCurrency)
            .eq(quoteCurrency != null, CryptoPrice::getQuoteCurrency, quoteCurrency)
            .eq(source != null, CryptoPrice::getSource, source)
            .orderByDesc(CryptoPrice::getTimestamp);

        Page<CryptoPrice> page = new Page<>(pageNum, pageSize);
        Page<CryptoPrice> result = cryptoPriceRepository.selectPage(page, wrapper);

        return ApiResponse.success(result);
    }

    @GetMapping("/crypto-price/latest")
    public ApiResponse<List<CryptoPrice>> getLatestPrices(
            @RequestParam(required = false) String baseCurrency,
            @RequestParam(required = false) String quoteCurrency) {
        List<CryptoPrice> prices;
        if (baseCurrency != null && quoteCurrency != null) {
            prices = cryptoPriceRepository.findByBaseCurrencyAndQuoteCurrency(baseCurrency, quoteCurrency);
        } else {
            prices = cryptoPriceRepository.selectList(null);
        }
        return ApiResponse.success(prices);
    }

    @GetMapping("/exchange-order/list")
    public ApiResponse<Page<ExchangeOrder>> listExchangeOrders(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) Long userId,
            @RequestParam(required = false) String orderNo,
            @RequestParam(required = false) String status,
            @RequestParam(required = false) String orderType) {

        LambdaQueryWrapper<ExchangeOrder> wrapper = new LambdaQueryWrapper<ExchangeOrder>()
            .eq(userId != null, ExchangeOrder::getUserId, userId)
            .like(orderNo != null, ExchangeOrder::getOrderNo, orderNo)
            .eq(status != null, ExchangeOrder::getStatus, status)
            .eq(orderType != null, ExchangeOrder::getOrderType, orderType)
            .orderByDesc(ExchangeOrder::getCreatedAt);

        Page<ExchangeOrder> page = new Page<>(pageNum, pageSize);
        Page<ExchangeOrder> result = exchangeOrderRepository.selectPage(page, wrapper);

        return ApiResponse.success(result);
    }

    @GetMapping("/exchange-order/{id}")
    public ApiResponse<ExchangeOrder> getExchangeOrderById(@PathVariable Long id) {
        ExchangeOrder order = exchangeOrderRepository.selectById(id);
        if (order != null) {
            return ApiResponse.success(order);
        }
        return ApiResponse.error(ResponseStatus.NOT_FOUND, "Exchange order not found");
    }

    @GetMapping("/exchange-order/statistics")
    public ApiResponse<Map<String, Object>> getExchangeOrderStatistics() {
        Long total = exchangeOrderRepository.selectCount(null);

        LambdaQueryWrapper<ExchangeOrder> pendingWrapper = new LambdaQueryWrapper<ExchangeOrder>()
            .eq(ExchangeOrder::getStatus, ExchangeOrder.OrderStatus.PENDING);
        Long pending = exchangeOrderRepository.selectCount(pendingWrapper);

        LambdaQueryWrapper<ExchangeOrder> completedWrapper = new LambdaQueryWrapper<ExchangeOrder>()
            .eq(ExchangeOrder::getStatus, ExchangeOrder.OrderStatus.COMPLETED);
        Long completed = exchangeOrderRepository.selectCount(completedWrapper);

        LambdaQueryWrapper<ExchangeOrder> failedWrapper = new LambdaQueryWrapper<ExchangeOrder>()
            .eq(ExchangeOrder::getStatus, ExchangeOrder.OrderStatus.FAILED);
        Long failed = exchangeOrderRepository.selectCount(failedWrapper);

        Map<String, Object> data = new HashMap<>();
        data.put("total", total);
        data.put("pending", pending);
        data.put("completed", completed);
        data.put("failed", failed);
        return ApiResponse.success(data);
    }

    @GetMapping("/statistics")
    public ApiResponse<Map<String, Object>> getPaymentStatistics() {
        Map<String, Object> data = new HashMap<>();

        Long totalKYC = userKYCRepository.selectCount(null);
        LambdaQueryWrapper<UserKYC> pendingKYCWrapper = new LambdaQueryWrapper<UserKYC>()
            .eq(UserKYC::getStatus, UserKYC.KYCStatus.PENDING);
        Long pendingKYC = userKYCRepository.selectCount(pendingKYCWrapper);

        Long totalOrders = exchangeOrderRepository.selectCount(null);
        LambdaQueryWrapper<ExchangeOrder> completedWrapper = new LambdaQueryWrapper<ExchangeOrder>()
            .eq(ExchangeOrder::getStatus, ExchangeOrder.OrderStatus.COMPLETED);
        Long completedOrders = exchangeOrderRepository.selectCount(completedWrapper);

        List<String> symbols = cryptoPriceRepository.findAllSymbols();

        data.put("totalKYC", totalKYC);
        data.put("pendingKYC", pendingKYC);
        data.put("totalOrders", totalOrders);
        data.put("completedOrders", completedOrders);
        data.put("cryptoSymbols", symbols.size());

        return ApiResponse.success(data);
    }
}