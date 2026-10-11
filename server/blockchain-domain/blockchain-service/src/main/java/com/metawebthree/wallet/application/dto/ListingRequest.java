package com.metawebthree.wallet.application.dto;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(description = "Request to create a marketplace listing")
public class ListingRequest {

    @Schema(description = "Seller wallet address")
    private String sellerAddress;

    @Schema(description = "Token mint address to sell")
    private String mintAddress;

    @Schema(description = "Payment token mint address (e.g. USDC)")
    private String paymentMintAddress;

    @Schema(description = "Price in payment token smallest units")
    private Long price;

    @Schema(description = "Amount of tokens to list (1 for NFT)")
    private Long listedAmount;

    public String getSellerAddress() { return sellerAddress; }
    public void setSellerAddress(String sellerAddress) { this.sellerAddress = sellerAddress; }
    public String getMintAddress() { return mintAddress; }
    public void setMintAddress(String mintAddress) { this.mintAddress = mintAddress; }
    public String getPaymentMintAddress() { return paymentMintAddress; }
    public void setPaymentMintAddress(String paymentMintAddress) { this.paymentMintAddress = paymentMintAddress; }
    public Long getPrice() { return price; }
    public void setPrice(Long price) { this.price = price; }
    public Long getListedAmount() { return listedAmount; }
    public void setListedAmount(Long listedAmount) { this.listedAmount = listedAmount; }
}
