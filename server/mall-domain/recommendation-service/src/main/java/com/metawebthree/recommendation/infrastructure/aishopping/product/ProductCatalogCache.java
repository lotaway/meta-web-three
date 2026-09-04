package com.metawebthree.recommendation.infrastructure.aishopping.product;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.springframework.stereotype.Component;

/**
 * Product catalog cache: refreshed during index building, used at match time to
 * restore vector hits into product information.
 */
@Component
public class ProductCatalogCache {

    private final ProductDataProvider productDataProvider;
    private final Map<Long, ProductDataProvider.ProductItem> byId = new ConcurrentHashMap<>();

    public ProductCatalogCache(ProductDataProvider productDataProvider) {
        this.productDataProvider = productDataProvider;
    }

    public synchronized void refresh() {
        List<ProductDataProvider.ProductItem> items = productDataProvider.fetchAllProducts();
        Map<Long, ProductDataProvider.ProductItem> next = new ConcurrentHashMap<>();
        for (ProductDataProvider.ProductItem item : items) {
            next.put(item.id, item);
        }
        byId.clear();
        byId.putAll(next);
    }

    public ProductDataProvider.ProductItem get(Long productId) {
        if (byId.isEmpty()) {
            refresh();
        }
        return byId.get(productId);
    }

    public List<ProductDataProvider.ProductItem> all() {
        if (byId.isEmpty()) {
            refresh();
        }
        return List.copyOf(byId.values());
    }

    public int size() {
        return byId.size();
    }

    public void clear() {
        byId.clear();
    }
}
