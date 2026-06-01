package com.metawebthree.product_recommendation.infrastructure.rpc;

import com.metawebthree.product_recommendation.application.dto.ProductProfileDTO;

import java.util.ArrayList;
import java.util.List;

public class ProductServiceClient {

    public List<ProductProfileDTO> getAllProductProfiles() {
        return new ArrayList<>();
    }

    public ProductProfileDTO getProductProfile(Long productId) {
        return null;
    }
}