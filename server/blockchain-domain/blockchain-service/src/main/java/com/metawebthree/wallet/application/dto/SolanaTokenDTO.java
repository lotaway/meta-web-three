package com.metawebthree.wallet.application.dto;

import io.swagger.v3.oas.annotations.media.Schema;

@Schema(description = "Solana token information")
public class SolanaTokenDTO {

    @Schema(description = "Mint address")
    private String mintAddress;

    @Schema(description = "Token name")
    private String name;

    @Schema(description = "Token symbol")
    private String symbol;

    @Schema(description = "Metadata URI")
    private String uri;

    @Schema(description = "Token type: TOKEN / NFT / SFT")
    private String tokenType;

    @Schema(description = "Token decimals")
    private int decimals;

    @Schema(description = "Total supply (raw amount)")
    private String supply;

    @Schema(description = "Owner address")
    private String owner;

    @Schema(description = "Transaction signature for creation")
    private String txSignature;

    public SolanaTokenDTO() {}

    public SolanaTokenDTO(String mintAddress, String name, String symbol, String uri,
                          String tokenType, int decimals, String supply,
                          String owner, String txSignature) {
        this.mintAddress = mintAddress;
        this.name = name;
        this.symbol = symbol;
        this.uri = uri;
        this.tokenType = tokenType;
        this.decimals = decimals;
        this.supply = supply;
        this.owner = owner;
        this.txSignature = txSignature;
    }

    public String getMintAddress() { return mintAddress; }
    public void setMintAddress(String mintAddress) { this.mintAddress = mintAddress; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getSymbol() { return symbol; }
    public void setSymbol(String symbol) { this.symbol = symbol; }
    public String getUri() { return uri; }
    public void setUri(String uri) { this.uri = uri; }
    public String getTokenType() { return tokenType; }
    public void setTokenType(String tokenType) { this.tokenType = tokenType; }
    public int getDecimals() { return decimals; }
    public void setDecimals(int decimals) { this.decimals = decimals; }
    public String getSupply() { return supply; }
    public void setSupply(String supply) { this.supply = supply; }
    public String getOwner() { return owner; }
    public void setOwner(String owner) { this.owner = owner; }
    public String getTxSignature() { return txSignature; }
    public void setTxSignature(String txSignature) { this.txSignature = txSignature; }
}
