package com.metawebthree.wallet.application.dto;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(description = "Request to create a Solana token/NFT/SFT")
public class CreateTokenRequest {

    @Schema(description = "Token name", example = "MyToken")
    private String name;

    @Schema(description = "Token symbol", example = "MTK")
    private String symbol;

    @Schema(description = "Metadata URI (JSON)", example = "https://example.com/token.json")
    private String uri;

    @Schema(description = "Token type: TOKEN / NFT / SFT", example = "TOKEN")
    private String tokenType;

    @Schema(description = "Initial supply (for TOKEN/SFT)", example = "1000000")
    private Long supply;

    @Schema(description = "Owner wallet address on Solana")
    private String ownerAddress;

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getSymbol() { return symbol; }
    public void setSymbol(String symbol) { this.symbol = symbol; }
    public String getUri() { return uri; }
    public void setUri(String uri) { this.uri = uri; }
    public String getTokenType() { return tokenType; }
    public void setTokenType(String tokenType) { this.tokenType = tokenType; }
    public Long getSupply() { return supply; }
    public void setSupply(Long supply) { this.supply = supply; }
    public String getOwnerAddress() { return ownerAddress; }
    public void setOwnerAddress(String ownerAddress) { this.ownerAddress = ownerAddress; }
}
