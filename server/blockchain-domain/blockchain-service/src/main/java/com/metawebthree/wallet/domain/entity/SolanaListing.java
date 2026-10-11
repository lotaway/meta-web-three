package com.metawebthree.wallet.domain.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import java.time.LocalDateTime;

@TableName("tb_solana_listing")
public class SolanaListing {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String listingAddress;
    private String sellerAddress;
    private String mintAddress;
    private String paymentMintAddress;
    private Long price;
    private Long listedAmount;
    private Integer status;
    private String txSignature;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public SolanaListing() {}

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getListingAddress() { return listingAddress; }
    public void setListingAddress(String listingAddress) { this.listingAddress = listingAddress; }
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
    public Integer getStatus() { return status; }
    public void setStatus(Integer status) { this.status = status; }
    public String getTxSignature() { return txSignature; }
    public void setTxSignature(String txSignature) { this.txSignature = txSignature; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}
