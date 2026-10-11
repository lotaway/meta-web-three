package com.metawebthree.wallet.interfaces.controller;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.wallet.application.dto.BuyRequest;
import com.metawebthree.wallet.application.dto.ListingDTO;
import com.metawebthree.wallet.application.dto.ListingRequest;
import com.metawebthree.wallet.application.service.SolanaMarketplaceService;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/solana/marketplace")
public class SolanaMarketplaceController {

    private final SolanaMarketplaceService marketplaceService;

    public SolanaMarketplaceController(SolanaMarketplaceService marketplaceService) {
        this.marketplaceService = marketplaceService;
    }

    @PostMapping("/listings")
    public ApiResponse<ListingDTO> createListing(@RequestBody ListingRequest request) {
        ListingDTO result = marketplaceService.createListing(request);
        return ApiResponse.success(result);
    }

    @GetMapping("/listings")
    public ApiResponse<List<ListingDTO>> listListings(
            @RequestParam(required = false) String seller) {
        List<ListingDTO> result = seller != null
            ? marketplaceService.listListingsBySeller(seller)
            : marketplaceService.listListings();
        return ApiResponse.success(result);
    }

    @GetMapping("/listings/{listingAddress}")
    public ApiResponse<ListingDTO> getListing(@PathVariable String listingAddress) {
        ListingDTO result = marketplaceService.getListing(listingAddress);
        return ApiResponse.success(result);
    }

    @PostMapping("/listings/{listingAddress}/buy")
    public ApiResponse<ListingDTO> buyGood(
            @PathVariable String listingAddress,
            @RequestBody Map<String, String> request) {
        BuyRequest buyRequest = new BuyRequest();
        buyRequest.setListingAddress(listingAddress);
        buyRequest.setBuyerAddress(request.get("buyerAddress"));
        buyRequest.setSellerAddress(request.get("sellerAddress"));
        buyRequest.setMintAddress(request.get("mintAddress"));
        ListingDTO result = marketplaceService.buyGood(buyRequest);
        return ApiResponse.success(result);
    }

    @PostMapping("/listings/{listingAddress}/delist")
    public ApiResponse<String> delistListing(
            @PathVariable String listingAddress,
            @RequestBody Map<String, String> request) {
        String sellerAddress = request.get("sellerAddress");
        String mintAddress = request.get("mintAddress");
        String result = marketplaceService.delistListing(listingAddress, sellerAddress, mintAddress);
        return ApiResponse.success(result);
    }
}
