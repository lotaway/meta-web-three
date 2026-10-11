package com.metawebthree.recommendation.domain.aishopping.service;

import com.metawebthree.recommendation.domain.aishopping.entity.AiProductMatch;
import com.metawebthree.recommendation.infrastructure.aishopping.provider.AiProviderClient;
import java.util.List;
import org.springframework.stereotype.Service;

/**
 * Smart match: embeds the natural-language query text and performs semantic
 * retrieval against the product text vector store.
 */
@Service
public class SmartMatchService {

    private final AiProviderClient providerClient;
    private final AiShoppingVectorSearch vectorSearch;

    public SmartMatchService(AiProviderClient providerClient, AiShoppingVectorSearch vectorSearch) {
        this.providerClient = providerClient;
        this.vectorSearch = vectorSearch;
    }

    public List<AiProductMatch> match(String query, int topK) {
        float[] vector = providerClient.embedText(query);
        return vectorSearch.searchText(vector, topK);
    }
}
