package com.metawebthree.wallet.application.dto;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(description = "Request to buy a listed good")
public class BuyRequest {

    @Schema(description = "Listing PDA address")
    private String listingAddress;

    @Schema(description = "Buyer wallet address")
    private String buyerAddress;

    @Schema(description = "Seller wallet address")
    private String sellerAddress;

    @Schema(description = "Token mint address being purchased")
    private String mintAddress;

    public String getListingAddress() { return listingAddress; }
    public void setListingAddress(String listingAddress) { this.listingAddress = listingAddress; }
    public String getBuyerAddress() { return buyerAddress; }
    public void setBuyerAddress(String buyerAddress) { this.buyerAddress = buyerAddress; }
    public String getSellerAddress() { return sellerAddress; }
    public void setSellerAddress(String sellerAddress) { this.sellerAddress = sellerAddress; }
    public String getMintAddress() { return mintAddress; }
    public void setMintAddress(String mintAddress) { this.mintAddress = mintAddress; }
}
