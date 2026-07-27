package com.metawebthree.wallet.application.dto;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(description = "Request to mint additional tokens")
public class MintTokenRequest {

    @Schema(description = "Token mint address")
    private String mintAddress;

    @Schema(description = "Recipient address")
    private String recipient;

    @Schema(description = "Amount to mint (raw units)")
    private Long amount;

    public String getMintAddress() { return mintAddress; }
    public void setMintAddress(String mintAddress) { this.mintAddress = mintAddress; }
    public String getRecipient() { return recipient; }
    public void setRecipient(String recipient) { this.recipient = recipient; }
    public Long getAmount() { return amount; }
    public void setAmount(Long amount) { this.amount = amount; }
}
