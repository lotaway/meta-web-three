package com.metawebthree.wallet.application.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.wallet.application.dto.BuyRequest;
import com.metawebthree.wallet.application.dto.ListingDTO;
import com.metawebthree.wallet.application.dto.ListingRequest;
import com.metawebthree.wallet.domain.entity.SolanaListing;
import com.metawebthree.wallet.infrastructure.persistence.repository.SolanaListingMapper;
import com.metawebthree.wallet.infrastructure.solana.SolanaContractClient;
import com.metawebthree.wallet.infrastructure.solana.SolanaProperties;
import com.metawebthree.wallet.infrastructure.solana.SolanaWalletManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class SolanaMarketplaceService {

    private static final Logger log = LoggerFactory.getLogger(SolanaMarketplaceService.class);

    private final SolanaContractClient contractClient;
    private final SolanaListingMapper listingMapper;
    private final SolanaWalletManager walletManager;
    private final SolanaCommissionService commissionService;
    private final SolanaProperties solanaProperties;

    public SolanaMarketplaceService(SolanaContractClient contractClient, SolanaListingMapper listingMapper,
                                    SolanaWalletManager walletManager, SolanaCommissionService commissionService,
                                    SolanaProperties solanaProperties) {
        this.contractClient = contractClient;
        this.listingMapper = listingMapper;
        this.walletManager = walletManager;
        this.commissionService = commissionService;
        this.solanaProperties = solanaProperties;
    }

    public ListingDTO createListing(ListingRequest request) {
        if (request.getSellerAddress() == null || request.getMintAddress() == null) {
            throw new IllegalArgumentException("Seller and mint address are required");
        }
        if (request.getPrice() == null || request.getPrice() <= 0) {
            throw new IllegalArgumentException("Price must be positive");
        }
        if (request.getListedAmount() == null || request.getListedAmount() <= 0) {
            throw new IllegalArgumentException("Listed amount must be positive");
        }

        byte[] seller = SolanaContractClient.decodeBase58(request.getSellerAddress());
        byte[] mint = SolanaContractClient.decodeBase58(request.getMintAddress());
        byte[] paymentMint = SolanaContractClient.decodeBase58(
            request.getPaymentMintAddress() != null ? request.getPaymentMintAddress() : solanaProperties.getPaymentMint());
        byte[] privateKey = walletManager.getPrivateKey(request.getSellerAddress());

        byte[] listing = contractClient.deriveListingAddress(seller, mint);
        String listingAddress = SolanaContractClient.base58Encode(listing);
        String paymentMintAddress = SolanaContractClient.base58Encode(paymentMint);
        String txSig = contractClient.listGood(seller, mint, paymentMint,
            request.getPrice(), request.getListedAmount(), privateKey);

        SolanaListing entity = new SolanaListing();
        entity.setListingAddress(listingAddress);
        entity.setSellerAddress(request.getSellerAddress());
        entity.setMintAddress(request.getMintAddress());
        entity.setPaymentMintAddress(paymentMintAddress);
        entity.setPrice(request.getPrice());
        entity.setListedAmount(request.getListedAmount());
        entity.setStatus(0);
        entity.setTxSignature(txSig);
        entity.setCreatedAt(LocalDateTime.now());
        entity.setUpdatedAt(LocalDateTime.now());
        listingMapper.insert(entity);

        return toDTO(entity);
    }

    public List<ListingDTO> listListings() {
        return listingMapper.selectList(null).stream()
            .map(this::toDTO)
            .collect(Collectors.toList());
    }

    public List<ListingDTO> listListingsBySeller(String sellerAddress) {
        return listingMapper.selectList(
                new LambdaQueryWrapper<SolanaListing>()
                    .eq(SolanaListing::getSellerAddress, sellerAddress))
            .stream()
            .map(this::toDTO)
            .collect(Collectors.toList());
    }

    public ListingDTO getListing(String listingAddress) {
        SolanaListing entity = listingMapper.selectOne(
            new LambdaQueryWrapper<SolanaListing>()
                .eq(SolanaListing::getListingAddress, listingAddress));
        return entity != null ? toDTO(entity) : null;
    }

    public ListingDTO buyGood(BuyRequest request) {
        byte[] buyer = SolanaContractClient.decodeBase58(request.getBuyerAddress());
        byte[] seller = SolanaContractClient.decodeBase58(request.getSellerAddress());
        byte[] mint = SolanaContractClient.decodeBase58(request.getMintAddress());
        byte[] paymentMint = SolanaContractClient.decodeBase58(solanaProperties.getPaymentMint());
        byte[] listing = SolanaContractClient.decodeBase58(request.getListingAddress());
        byte[] privateKey = walletManager.getPrivateKey(request.getBuyerAddress());

        String txSig = contractClient.buyGood(buyer, seller, mint, paymentMint, listing, privateKey);

        SolanaListing entity = listingMapper.selectOne(
            new LambdaQueryWrapper<SolanaListing>()
                .eq(SolanaListing::getListingAddress, request.getListingAddress()));
        long saleAmount = entity != null && entity.getPrice() != null ? entity.getPrice() : 0L;
        if (entity != null) {
            entity.setStatus(1);
            entity.setTxSignature(txSig);
            entity.setUpdatedAt(LocalDateTime.now());
            listingMapper.updateById(entity);
        }

        // Optional commission linkage: only when the seller actually has an
        // upline and the backend custodies the seller wallet. Best-effort — a
        // commission failure must not turn a completed purchase into a failure.
        if (saleAmount > 0 && commissionService.getUplineAddress(request.getSellerAddress()) != null) {
            try {
                commissionService.distributeCommission(request.getSellerAddress(), solanaProperties.getPaymentMint(), saleAmount);
            } catch (Exception ex) {
                log.warn("Auto-commission failed after purchase {} for seller {}: {}",
                    request.getListingAddress(), request.getSellerAddress(), ex.getMessage());
            }
        }

        return new ListingDTO(
            request.getListingAddress(), request.getSellerAddress(), request.getMintAddress(),
            solanaProperties.getPaymentMint(), 0L, 0L, 1,
            System.currentTimeMillis() / 1000, txSig
        );
    }

    public String delistListing(String listingAddress, String sellerAddress, String mintAddress) {
        SolanaListing entity = listingMapper.selectOne(
            new LambdaQueryWrapper<SolanaListing>()
                .eq(SolanaListing::getListingAddress, listingAddress));
        String paymentMintAddress = entity != null && entity.getPaymentMintAddress() != null
            && !entity.getPaymentMintAddress().isBlank()
            ? entity.getPaymentMintAddress()
            : solanaProperties.getPaymentMint();

        byte[] seller = SolanaContractClient.decodeBase58(sellerAddress);
        byte[] mint = SolanaContractClient.decodeBase58(mintAddress);
        byte[] paymentMint = SolanaContractClient.decodeBase58(paymentMintAddress);
        byte[] privateKey = walletManager.getPrivateKey(sellerAddress);

        String txSig = contractClient.delistGood(seller, mint, paymentMint, privateKey);

        if (entity != null) {
            entity.setStatus(2);
            entity.setTxSignature(txSig);
            entity.setUpdatedAt(LocalDateTime.now());
            listingMapper.updateById(entity);
        }

        return txSig;
    }

    private ListingDTO toDTO(SolanaListing entity) {
        return new ListingDTO(
            entity.getListingAddress(),
            entity.getSellerAddress(),
            entity.getMintAddress(),
            entity.getPaymentMintAddress(),
            entity.getPrice(),
            entity.getListedAmount(),
            entity.getStatus(),
            entity.getCreatedAt() != null ? entity.getCreatedAt().atZone(java.time.ZoneOffset.UTC).toEpochSecond() : 0L,
            entity.getTxSignature()
        );
    }
}
