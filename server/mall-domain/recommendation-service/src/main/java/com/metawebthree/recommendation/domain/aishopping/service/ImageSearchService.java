package com.metawebthree.recommendation.domain.aishopping.service;

import com.metawebthree.recommendation.domain.aishopping.entity.AiProductMatch;
import com.metawebthree.recommendation.infrastructure.aishopping.provider.AiProviderClient;
import java.util.List;
import org.springframework.stereotype.Service;

/**
 * Image search: embeds the uploaded user image and retrieves similar products
 * from the product image vector store.
 */
@Service
public class ImageSearchService {

    private final AiProviderClient providerClient;
    private final AiShoppingVectorSearch vectorSearch;

    public ImageSearchService(AiProviderClient providerClient, AiShoppingVectorSearch vectorSearch) {
        this.providerClient = providerClient;
        this.vectorSearch = vectorSearch;
    }

    public List<AiProductMatch> search(byte[] imageBytes, int topK) {
        float[] vector = providerClient.embedImage(imageBytes);
        return vectorSearch.searchImage(vector, topK);
    }
}
