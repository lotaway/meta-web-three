package com.metawebthree.payment.interfaces.web;

import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.payment.application.PaymentService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletRequest;
import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/pay")
@RequiredArgsConstructor
@Slf4j
public class PayController {

    private final PaymentService paymentService;

    @PostMapping("/wechat/params")
    public ApiResponse<Map<String, String>> getWechatParams(
            @RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestBody Map<String, Long> body) {
        
        Long orderId = body.get("orderId");
        if (orderId == null) {
            return ApiResponse.error(ResponseStatus.PARAM_MISSING_ERROR);
        }
        
        try {
            Map<String, String> params = paymentService.getWechatPayParams(orderId, userId);
            return ApiResponse.success(params);
        } catch (Exception e) {
            log.error("Get wechat params failed", e);
            return ApiResponse.error(ResponseStatus.SYSTEM_ERROR);
        }
    }

    @PostMapping("/alipay/params")
    public ApiResponse<Map<String, String>> getAlipayParams(
            @RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestBody Map<String, Long> body) {
        
        Long orderId = body.get("orderId");
        if (orderId == null) {
            return ApiResponse.error(ResponseStatus.PARAM_MISSING_ERROR);
        }
        
        try {
            Map<String, String> params = paymentService.getAlipayParams(orderId, userId);
            return ApiResponse.success(params);
        } catch (Exception e) {
            log.error("Get alipay params failed", e);
            return ApiResponse.error(ResponseStatus.SYSTEM_ERROR);
        }
    }

    @PostMapping("/stripe/params")
    public ApiResponse<Map<String, String>> getStripeParams(
            @RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestBody Map<String, Long> body) {
        
        Long orderId = body.get("orderId");
        if (orderId == null) {
            return ApiResponse.error(ResponseStatus.PARAM_MISSING_ERROR);
        }
        
        try {
            Map<String, String> params = paymentService.getStripeParams(orderId, userId);
            return ApiResponse.success(params);
        } catch (Exception e) {
            log.error("Get stripe params failed", e);
            return ApiResponse.error(ResponseStatus.SYSTEM_ERROR);
        }
    }

    @PostMapping("/verify")
    public ApiResponse<Map<String, Object>> verifyPayment(
            @RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestBody Map<String, String> body) {
        
        String orderId = body.get("orderId");
        String transactionId = body.get("transactionId");

        if (orderId == null || transactionId == null) {
            return ApiResponse.error(ResponseStatus.PARAM_MISSING_ERROR);
        }

        try {
            boolean valid = paymentService.verifyPayment(orderId, transactionId, userId);
            return ApiResponse.success(Map.of("valid", valid));
        } catch (Exception e) {
            log.error("Verify payment failed", e);
            return ApiResponse.error(ResponseStatus.SYSTEM_ERROR);
        }
    }

    @GetMapping("/alipay/query")
    public ApiResponse<Map<String, Object>> queryAlipayStatus(
            @RequestHeader(HeaderConstants.USER_ID) Long userId,
            @RequestParam String outTradeNo) {
        
        if (outTradeNo == null) {
            return ApiResponse.error(ResponseStatus.PARAM_MISSING_ERROR);
        }

        try {
            Map<String, Object> result = paymentService.queryAlipayStatus(outTradeNo, userId);
            return ApiResponse.success(result);
        } catch (Exception e) {
            log.error("Query alipay status failed", e);
            return ApiResponse.error(ResponseStatus.SYSTEM_ERROR);
        }
    }

    /**
     * 支付宝异步回调接口
     * 必须为POST请求
     * 支付宝文档：https://opendocs.alipay.com/open/270/105902
     */
    @Operation(summary = "支付宝异步回调", description = "处理支付宝支付异步通知回调")
    @PostMapping("/alipay/callback")
    public String alipayCallback(HttpServletRequest request) {
        // 获取所有请求参数
        Map<String, String> params = new HashMap<>();
        Map<String, String[]> requestParams = request.getParameterMap();
        for (String name : requestParams.keySet()) {
            params.put(name, request.getParameter(name));
        }
        
        log.info("Alipay callback received: {}", params);
        
        try {
            String result = paymentService.handleAlipayCallback(params);
            return result;
        } catch (Exception e) {
            log.error("Alipay callback processing failed", e);
            return "failure";
        }
    }
}