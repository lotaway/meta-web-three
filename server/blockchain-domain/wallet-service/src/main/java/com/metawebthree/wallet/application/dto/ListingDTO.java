package com.metawebthree.wallet.application.dto;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(description = "Marketplace listing information")
public class ListingDTO {

    @Schema(description = "Listing PDA address")
    private String listingAddress;

    @Schema(description = "Seller wallet address")
    private String seller;

    @Schema(description = "Token mint address being sold")
    private String mint;

    @Schema(description = "Payment token mint address")
    private String paymentMint;

    @Schema(description = "Price in payment token smallest units")
    private Long price;

    @Schema(description = "Amount of tokens listed (1 for NFT)")
    private Long listedAmount;

    @Schema(description = "Status: 0=Active, 1=Sold, 2=Cancelled")
    private Integer status;

    @Schema(description = "Creation timestamp")
    private Long createdAt;

    @Schema(description = "Transaction signature")
    private String txSignature;

    public ListingDTO() {}

    public ListingDTO(String listingAddress, String seller, String mint, String paymentMint,
                      Long price, Long listedAmount, Integer status, Long createdAt, String txSignature) {
        this.listingAddress = listingAddress;
        this.seller = seller;
        this.mint = mint;
        this.paymentMint = paymentMint;
        this.price = price;
        this.listedAmount = listedAmount;
        this.status = status;
        this.createdAt = createdAt;
        this.txSignature = txSignature;
    }

    public String getListingAddress() { return listingAddress; }
    public void setListingAddress(String listingAddress) { this.listingAddress = listingAddress; }
    public String getSeller() { return seller; }
    public void setSeller(String seller) { this.seller = seller; }
    public String getMint() { return mint; }
    public void setMint(String mint) { this.mint = mint; }
    public String getPaymentMint() { return paymentMint; }
    public void setPaymentMint(String paymentMint) { this.paymentMint = paymentMint; }
    public Long getPrice() { return price; }
    public void setPrice(Long price) { this.price = price; }
    public Long getListedAmount() { return listedAmount; }
    public void setListedAmount(Long listedAmount) { this.listedAmount = listedAmount; }
    public Integer getStatus() { return status; }
    public void setStatus(Integer status) { this.status = status; }
    public Long getCreatedAt() { return createdAt; }
    public void setCreatedAt(Long createdAt) { this.createdAt = createdAt; }
    public String getTxSignature() { return txSignature; }
    public void setTxSignature(String txSignature) { this.txSignature = txSignature; }
}
