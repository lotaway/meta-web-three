package com.metawebthree.wallet.interfaces.controller;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.wallet.application.dto.CouponDTO;
import com.metawebthree.wallet.application.service.SolanaCouponService;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/solana/coupons")
public class SolanaCouponController {

    private final SolanaCouponService couponService;

    public SolanaCouponController(SolanaCouponService couponService) {
        this.couponService = couponService;
    }

    @PostMapping
    public ApiResponse<CouponDTO> createCoupon(@RequestBody Map<String, Object> request) {
        String authority = (String) request.get("authority");
        String mint = (String) request.get("mint");
        Long discountAmount = request.get("discountAmount") != null ? ((Number) request.get("discountAmount")).longValue() : null;
        Long maxUses = request.get("maxUses") != null ? ((Number) request.get("maxUses")).longValue() : null;
        Long expiry = request.get("expiry") != null ? ((Number) request.get("expiry")).longValue() : null;
        CouponDTO result = couponService.createCoupon(authority, mint, discountAmount, maxUses, expiry);
        return ApiResponse.success(result);
    }

    @PostMapping("/redeem")
    public ApiResponse<CouponDTO> redeemCoupon(@RequestBody Map<String, Object> request) {
        String authority = (String) request.get("authority");
        String user = (String) request.get("user");
        String mint = (String) request.get("mint");
        @SuppressWarnings("unchecked")
        List<String> proof = (List<String>) request.get("proof");
        CouponDTO result = couponService.redeemCoupon(authority, user, mint, proof);
        return ApiResponse.success(result);
    }
}
