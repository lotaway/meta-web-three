package com.metawebthree.wallet.application.dto;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(description = "Coupon information")
public class CouponDTO {

    @Schema(description = "Coupon PDA address")
    private String couponAddress;

    @Schema(description = "Authority wallet address")
    private String authority;

    @Schema(description = "Token mint address")
    private String mint;

    @Schema(description = "Discount amount per redemption")
    private Long discountAmount;

    @Schema(description = "Maximum number of uses")
    private Long maxUses;

    @Schema(description = "Total redeemed count")
    private Long totalRedeemed;

    @Schema(description = "Expiry timestamp")
    private Long expiry;

    @Schema(description = "Transaction signature")
    private String txSignature;

    public CouponDTO() {}

    public CouponDTO(String couponAddress, String authority, String mint, Long discountAmount,
                     Long maxUses, Long totalRedeemed, Long expiry, String txSignature) {
        this.couponAddress = couponAddress;
        this.authority = authority;
        this.mint = mint;
        this.discountAmount = discountAmount;
        this.maxUses = maxUses;
        this.totalRedeemed = totalRedeemed;
        this.expiry = expiry;
        this.txSignature = txSignature;
    }

    public String getCouponAddress() { return couponAddress; }
    public void setCouponAddress(String couponAddress) { this.couponAddress = couponAddress; }
    public String getAuthority() { return authority; }
    public void setAuthority(String authority) { this.authority = authority; }
    public String getMint() { return mint; }
    public void setMint(String mint) { this.mint = mint; }
    public Long getDiscountAmount() { return discountAmount; }
    public void setDiscountAmount(Long discountAmount) { this.discountAmount = discountAmount; }
    public Long getMaxUses() { return maxUses; }
    public void setMaxUses(Long maxUses) { this.maxUses = maxUses; }
    public Long getTotalRedeemed() { return totalRedeemed; }
    public void setTotalRedeemed(Long totalRedeemed) { this.totalRedeemed = totalRedeemed; }
    public Long getExpiry() { return expiry; }
    public void setExpiry(Long expiry) { this.expiry = expiry; }
    public String getTxSignature() { return txSignature; }
    public void setTxSignature(String txSignature) { this.txSignature = txSignature; }
}
