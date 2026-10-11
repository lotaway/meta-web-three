package com.metawebthree.recommendation.domain.aishopping.service;

import com.metawebthree.recommendation.application.aishopping.AiProviderConfig;
import com.metawebthree.recommendation.application.aishopping.AiProviderSettings;
import com.metawebthree.recommendation.domain.aishopping.entity.AiProductMatch;
import com.metawebthree.recommendation.infrastructure.aishopping.product.ProductCatalogCache;
import com.metawebthree.recommendation.infrastructure.aishopping.product.ProductDataProvider;
import com.metawebthree.recommendation.infrastructure.aishopping.vector.VectorHit;
import com.metawebthree.recommendation.infrastructure.aishopping.vector.VectorStore;
import com.metawebthree.recommendation.infrastructure.aishopping.vector.VectorStoreFactory;
import java.util.ArrayList;
import java.util.List;
import org.springframework.stereotype.Service;

/**
 * Shared vector search capability: queries the vector store and restores hits
 * into product match results.
 */
@Service
public class AiShoppingVectorSearch {

    private final VectorStoreFactory vectorStoreFactory;
    private final ProductCatalogCache catalogCache;
    private final AiProviderConfig providerConfig;

    public AiShoppingVectorSearch(VectorStoreFactory vectorStoreFactory,
                                  ProductCatalogCache catalogCache,
                                  AiProviderConfig providerConfig) {
        this.vectorStoreFactory = vectorStoreFactory;
        this.catalogCache = catalogCache;
        this.providerConfig = providerConfig;
    }

    public List<AiProductMatch> searchText(float[] queryVector, int topK) {
        AiProviderSettings settings = providerConfig.getSettings();
        return search(settings.getMilvusCollectionText(), queryVector, topK, "Smart match");
    }

    public List<AiProductMatch> searchImage(float[] imageVector, int topK) {
        AiProviderSettings settings = providerConfig.getSettings();
        return search(settings.getMilvusCollectionImage(), imageVector, topK, "Image search");
    }

    private List<AiProductMatch> search(String collection, float[] vector, int topK, String reasonPrefix) {
        VectorStore store = vectorStoreFactory.getStore();
        store.ensureCollection(collection, providerConfig.getSettings().getEmbeddingDim());

        List<VectorHit> hits = store.search(collection, vector, topK);
        List<AiProductMatch> matches = new ArrayList<>();
        for (VectorHit hit : hits) {
            ProductDataProvider.ProductItem product = catalogCache.get(hit.getProductId());
            if (product == null) {
                continue;
            }
            String price = product.price > 0 ? String.valueOf(product.price) : null;
            String reason = reasonPrefix + " similarity " + String.format("%.2f", hit.getScore());
            matches.add(new AiProductMatch(
                    product.id, product.name, product.pic, price,
                    (double) hit.getScore(), reason));
        }
        return matches;
    }
}
