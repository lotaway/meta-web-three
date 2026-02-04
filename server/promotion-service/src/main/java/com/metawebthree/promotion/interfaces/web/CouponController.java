package com.metawebthree.promotion.interfaces.web;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import jakarta.servlet.http.HttpServletRequest;

import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.metawebthree.common.constants.RequestHeaderKeys;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.promotion.application.CouponCommandService;
import com.metawebthree.promotion.application.CouponQueryService;
import com.metawebthree.promotion.application.CouponTypeCommandService;
import com.metawebthree.promotion.application.CouponTypeQueryService;
import com.metawebthree.promotion.domain.exception.PromotionErrorCode;
import com.metawebthree.promotion.domain.exception.PromotionException;
import com.metawebthree.promotion.domain.model.Coupon;
import com.metawebthree.promotion.domain.model.CouponType;
import com.metawebthree.promotion.interfaces.web.dto.CouponAssignRequest;
import com.metawebthree.promotion.interfaces.web.dto.CouponBatchAssignRequest;
import com.metawebthree.promotion.interfaces.web.dto.CouponBatchView;
import com.metawebthree.promotion.interfaces.web.dto.CouponClaimRequest;
import com.metawebthree.promotion.interfaces.web.dto.CouponConsumeRequest;
import com.metawebthree.promotion.interfaces.web.dto.CouponGenerateRequest;
import com.metawebthree.promotion.interfaces.web.dto.CouponTransferRequest;
import com.metawebthree.promotion.interfaces.web.dto.CouponTypeCreateRequest;
import com.metawebthree.promotion.interfaces.web.dto.CouponValidateRequest;
import com.metawebthree.promotion.interfaces.web.dto.CouponValidateResponse;
import com.metawebthree.promotion.interfaces.web.dto.CouponView;

import jakarta.validation.Valid;

@Validated
@RestController
@RequestMapping("/v1/promotion")
public class CouponController {
    private final CouponTypeCommandService couponTypeCommandService;
    private final CouponTypeQueryService couponTypeQueryService;
    private final CouponCommandService couponCommandService;
    private final CouponQueryService couponQueryService;

    public CouponController(CouponTypeCommandService couponTypeCommandService,
            CouponTypeQueryService couponTypeQueryService, CouponCommandService couponCommandService,
            CouponQueryService couponQueryService) {
        this.couponTypeCommandService = couponTypeCommandService;
        this.couponTypeQueryService = couponTypeQueryService;
        this.couponCommandService = couponCommandService;
        this.couponQueryService = couponQueryService;
    }

    @GetMapping("/health")
    public ApiResponse<String> healthCheck() {
        return ApiResponse.success("promotion-service is running");
    }

    @PostMapping("/coupon-types")
    public ApiResponse<Void> createCouponType(
            HttpServletRequest httpRequest,
            @Valid @RequestBody CouponTypeCreateRequest request) {
        return executeCommand(() -> {
            CouponType type = toCouponType(request, readUserId(httpRequest));
            couponTypeCommandService.create(type);
        });
    }

    @PostMapping("/coupon-types/{id}/generate")
    public ApiResponse<Void> generateCoupons(@PathVariable Long id,
            @Valid @RequestBody CouponGenerateRequest request) {
        return executeCommand(() -> couponCommandService.generateBatch(request.getBatchId(), id, request.getCount()));
    }

    @GetMapping("/coupon-batches/{batchId}")
    public ApiResponse<CouponBatchView> getBatch(@PathVariable String batchId) {
        return executeQuery(() -> toBatchView(batchId, couponQueryService.listByBatch(batchId)));
    }

    @PostMapping("/coupons/assign")
    public ApiResponse<Void> assignCoupon(@Valid @RequestBody CouponAssignRequest request) {
        return executeCommand(() -> couponCommandService.assignCoupon(request.getCode(),
                request.getUserId()));
    }

    @PostMapping("/coupons/assign-batch")
    public ApiResponse<Void> assignCoupons(@Valid @RequestBody CouponBatchAssignRequest request) {
        return executeCommand(() -> couponCommandService.batchAssign(request.getCouponTypeId(),
                request.getUserIds(), request.getAmount()));
    }

    @PostMapping("/coupons/claim")
    public ApiResponse<Void> claimCoupon(
            HttpServletRequest httpRequest,
            @Valid @RequestBody CouponClaimRequest request) {
        return executeCommand(() -> couponCommandService.claim(request.getCouponTypeId(), readUserId(httpRequest)));
    }

    @GetMapping("/coupons")
    public ApiResponse<List<CouponView>> listCoupons(
            HttpServletRequest httpRequest,
            @RequestParam(required = false) Integer useStatus) {
        return executeQuery(() -> toCouponViews(readUserId(httpRequest), useStatus));
    }

    @PostMapping("/coupons/validate")
    public ApiResponse<CouponValidateResponse> validateCoupon(
            HttpServletRequest httpRequest,
            @Valid @RequestBody CouponValidateRequest request) {
        return executeQuery(() -> toValidateResponse(readUserId(httpRequest), request));
    }

    @PostMapping("/coupons/consume")
    public ApiResponse<Void> consumeCoupon(
            HttpServletRequest httpRequest,
            @Valid @RequestBody CouponConsumeRequest request) {
        return executeCommand(() -> couponCommandService.consume(request.getCode(),
                readUserId(httpRequest), request.getOrderNo(),
                prefer(readUserName(httpRequest), request.getConsumerName()), request.getOperatorName()));
    }

    @PostMapping("/coupons/offline-consume")
    public ApiResponse<Void> offlineConsume(@Valid @RequestBody CouponConsumeRequest request) {
        return executeCommand(() -> couponCommandService.consume(request.getCode(), null,
                request.getOrderNo(), request.getConsumerName(), request.getOperatorName()));
    }

    @PostMapping("/coupons/transfer/open")
    public ApiResponse<Void> openTransfer(
            HttpServletRequest httpRequest,
            @Valid @RequestBody CouponTransferRequest request) {
        return executeCommand(() -> couponCommandService.openTransfer(request.getCode(),
                readUserId(httpRequest)));
    }

    @PostMapping("/coupons/transfer/close")
    public ApiResponse<Void> closeTransfer(
            HttpServletRequest httpRequest,
            @Valid @RequestBody CouponTransferRequest request) {
        return executeCommand(() -> couponCommandService.closeTransfer(request.getCode(),
                readUserId(httpRequest)));
    }

    @PostMapping("/coupons/transfer/claim")
    public ApiResponse<Void> claimTransfer(
            HttpServletRequest httpRequest,
            @Valid @RequestBody CouponTransferRequest request) {
        return executeCommand(() -> couponCommandService.claimTransfer(request.getCode(),
                readUserId(httpRequest)));
    }

    private CouponType toCouponType(CouponTypeCreateRequest request, Long userId) {
        CouponType type = new CouponType();
        type.setName(request.getName());
        type.setDescription(request.getDescription());
        type.setImageUrl(request.getImageUrl());
        type.setMinimumOrderAmount(request.getMinimumOrderAmount());
        type.setDiscountAmount(request.getDiscountAmount());
        type.setStartTime(request.getStartTime());
        type.setEndTime(request.getEndTime());
        type.setIsEnabled(request.getIsEnabled());
        type.setTypeCode(request.getTypeCode());
        type.setCreateUserId(userId);
        return type;
    }

    private CouponBatchView toBatchView(String batchId, List<Coupon> coupons) {
        CouponBatchView view = new CouponBatchView();
        view.setBatchId(batchId);
        view.setCodes(extractCodes(coupons));
        return view;
    }

    private List<String> extractCodes(List<Coupon> coupons) {
        List<String> codes = new ArrayList<>();
        for (Coupon coupon : coupons) {
            codes.add(coupon.getCode());
        }
        return codes;
    }

    private List<CouponView> toCouponViews(Long userId, Integer useStatus) {
        List<Coupon> coupons = couponQueryService.listByOwner(userId, useStatus);
        List<CouponView> views = new ArrayList<>();
        for (Coupon coupon : coupons) {
            views.add(toCouponView(coupon));
        }
        return views;
    }

    private CouponView toCouponView(Coupon coupon) {
        CouponType type = couponTypeQueryService.getCouponType(coupon.getCouponTypeId());
        CouponView view = new CouponView();
        view.setCode(coupon.getCode());
        view.setCouponTypeId(coupon.getCouponTypeId());
        view.setUseStatus(coupon.getUseStatus());
        view.setTransferStatus(coupon.getTransferStatus());
        view.setAcquireMethod(coupon.getAcquireMethod());
        if (type != null) {
            view.setCouponTypeName(type.getName());
            view.setMinimumOrderAmount(type.getMinimumOrderAmount());
            view.setDiscountAmount(type.getDiscountAmount());
            view.setStartTime(type.getStartTime());
            view.setEndTime(type.getEndTime());
        }
        return view;
    }

    private CouponValidateResponse toValidateResponse(Long userId, CouponValidateRequest request) {
        CouponQueryService.CouponValidateResult result = couponQueryService.validate(
                request.getCode(), userId, request.getOrderAmount(), request.getDeliveryFee());
        CouponValidateResponse response = new CouponValidateResponse();
        response.setCouponTypeName(result.getCouponTypeName());
        response.setDiscountAmount(result.getDiscountAmount());
        response.setPayableAmount(result.getPayableAmount());
        return response;
    }

    private String prefer(String primary, String fallback) {
        return primary == null || primary.isBlank() ? fallback : primary;
    }

    private Long readUserId(HttpServletRequest request) {
        String userId = request.getHeader(RequestHeaderKeys.USER_ID.getValue());
        if (userId == null) {
            throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "missing user header");
        }
        try {
            return Long.parseLong(userId);
        } catch (NumberFormatException ex) {
            throw new PromotionException(PromotionErrorCode.INVALID_REQUEST, "invalid user header");
        }
    }

    private String readUserName(HttpServletRequest request) {
        return request.getHeader(RequestHeaderKeys.USER_NAME.getValue());
    }

    private ApiResponse<Void> executeCommand(Runnable command) {
        try {
            command.run();
            return ApiResponse.success();
        } catch (PromotionException ex) {
            return ApiResponse.error(ResponseStatus.PARAM_VALIDATION_ERROR, formatMessage(ex));
        }
    }

    private <T> ApiResponse<T> executeQuery(Supplier<T> query) {
        try {
            return ApiResponse.success(query.get());
        } catch (PromotionException ex) {
            return ApiResponse.error(ResponseStatus.PARAM_VALIDATION_ERROR, formatMessage(ex));
        }
    }

    private String formatMessage(PromotionException ex) {
        return ex.getErrorCode().getCode() + ":" + ex.getMessage();
    }
}
