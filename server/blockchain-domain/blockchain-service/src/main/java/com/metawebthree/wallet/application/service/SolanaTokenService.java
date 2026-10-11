package com.metawebthree.wallet.application.service;

import com.metawebthree.wallet.application.dto.CreateTokenRequest;
import com.metawebthree.wallet.application.dto.MintTokenRequest;
import com.metawebthree.wallet.application.dto.SolanaTokenDTO;
import com.metawebthree.wallet.infrastructure.solana.SolanaContractClient;
import com.metawebthree.wallet.infrastructure.solana.SolanaRpcClient;
import com.metawebthree.wallet.infrastructure.solana.SolanaWalletManager;
import org.springframework.stereotype.Service;

import java.math.BigInteger;
import java.util.List;
import java.util.Map;

@Service
public class SolanaTokenService {

    private final SolanaRpcClient solanaRpcClient;
    private final SolanaContractClient contractClient;
    private final SolanaWalletManager walletManager;

    public SolanaTokenService(SolanaRpcClient solanaRpcClient, SolanaContractClient contractClient, SolanaWalletManager walletManager) {
        this.solanaRpcClient = solanaRpcClient;
        this.contractClient = contractClient;
        this.walletManager = walletManager;
    }

    public SolanaTokenDTO createToken(CreateTokenRequest request) {
        String tokenType = request.getTokenType() != null ? request.getTokenType().toUpperCase() : "TOKEN";
        if (!List.of("TOKEN", "NFT", "SFT").contains(tokenType)) {
            throw new IllegalArgumentException("Invalid token type: " + tokenType);
        }

        long supply = request.getSupply() != null ? request.getSupply() : 0L;
        if ("TOKEN".equals(tokenType) && supply <= 0) {
            throw new IllegalArgumentException("Supply is required for TOKEN type");
        }
        if ("SFT".equals(tokenType) && supply <= 0) {
            throw new IllegalArgumentException("Supply is required for SFT type");
        }
        if ("NFT".equals(tokenType)) supply = 1L;

        byte[] authority = SolanaContractClient.decodeBase58(request.getOwnerAddress());
        byte[] privateKey = walletManager.getPrivateKey(request.getOwnerAddress());
        byte[] mint;
        String txSig;
        int decimals;

        if ("SFT".equals(tokenType) || "NFT".equals(tokenType)) {
            txSig = contractClient.createSft(request.getName(), request.getSymbol(),
                request.getUri() != null ? request.getUri() : "", supply, authority, privateKey);
            mint = contractClient.deriveSftMint(request.getName(), authority);
            decimals = 0;
        } else {
            var ti = new SolanaContractClient.TokenInstruction(request.getName(), request.getSymbol(),
                request.getUri() != null ? request.getUri() : "", supply, authority, null, null, null);
            txSig = contractClient.createToken(ti, privateKey);
            mint = contractClient.deriveTokenMint(request.getName(), authority);
            decimals = 9;
        }

        return new SolanaTokenDTO(
            SolanaContractClient.base58Encode(mint),
            request.getName(), request.getSymbol(), request.getUri(),
            tokenType, decimals, String.valueOf(supply),
            request.getOwnerAddress(), txSig
        );
    }

    @SuppressWarnings("unchecked")
    public SolanaTokenDTO getToken(String mintAddress) {
        Map<String, Object> accountInfo = solanaRpcClient.getAccountInfo(mintAddress);
        if (accountInfo.isEmpty()) {
            throw new IllegalArgumentException("Token not found: " + mintAddress);
        }
        Map<String, Object> data = (Map<String, Object>) accountInfo.get("data");
        Map<String, Object> parsed = (Map<String, Object>) data.get("parsed");
        Map<String, Object> info = (Map<String, Object>) parsed.get("info");

        int decimals = ((Number) info.get("decimals")).intValue();
        String supply = info.get("supply") != null ? info.get("supply").toString() : "0";
        String owner = info.get("authority") != null
            ? ((Map<String, String>) info.get("authority")).get("address") : "";

        String tokenType;
        if (decimals == 0) {
            BigInteger supplyBI = new BigInteger(supply);
            tokenType = supplyBI.compareTo(BigInteger.ONE) <= 0 ? "NFT" : "SFT";
        } else {
            tokenType = "TOKEN";
        }

        SolanaTokenDTO dto = new SolanaTokenDTO();
        dto.setMintAddress(mintAddress);
        dto.setTokenType(tokenType);
        dto.setDecimals(decimals);
        dto.setSupply(supply);
        dto.setOwner(owner);
        return dto;
    }

    public SolanaTokenDTO mintTo(MintTokenRequest request) {
        byte[] mint = SolanaContractClient.decodeBase58(request.getMintAddress());
        byte[] receiver = SolanaContractClient.decodeBase58(request.getRecipient());
        byte[] authority = receiver;
        byte[] privateKey = walletManager.getPrivateKey(request.getRecipient());

        String txSig = contractClient.mintTo(mint, receiver, request.getAmount(), authority, privateKey);

        SolanaTokenDTO dto = new SolanaTokenDTO();
        dto.setMintAddress(request.getMintAddress());
        dto.setSupply(String.valueOf(request.getAmount()));
        dto.setOwner(request.getRecipient());
        dto.setTxSignature(txSig);
        return dto;
    }

    public String burnTokens(String mintAddress, Long amount, String ownerAddress) {
        if (amount == null || amount <= 0) {
            throw new IllegalArgumentException("Amount must be positive");
        }
        byte[] mint = SolanaContractClient.decodeBase58(mintAddress);
        byte[] privateKey = walletManager.getPrivateKey(ownerAddress);
        return contractClient.burnTokens(mint, privateKey, amount, privateKey);
    }
}
