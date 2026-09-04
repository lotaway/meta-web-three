package com.metawebthree.recommendation.domain.aishopping.service;

import com.metawebthree.recommendation.application.aishopping.AiProviderConfig;
import com.metawebthree.recommendation.application.aishopping.AiProviderSettings;
import com.metawebthree.recommendation.application.aishopping.AiShoppingFeatureGuard;
import com.metawebthree.recommendation.domain.aishopping.entity.AiShoppingConfig;
import com.metawebthree.recommendation.domain.aishopping.entity.IndexStatus;
import com.metawebthree.recommendation.domain.aishopping.repository.AiShoppingConfigRepository;
import com.metawebthree.recommendation.infrastructure.aishopping.provider.AiProviderClient;
import com.metawebthree.recommendation.infrastructure.aishopping.product.ProductCatalogCache;
import com.metawebthree.recommendation.infrastructure.aishopping.product.ProductDataProvider;
import com.metawebthree.recommendation.infrastructure.aishopping.vector.VectorRecord;
import com.metawebthree.recommendation.infrastructure.aishopping.vector.VectorStore;
import com.metawebthree.recommendation.infrastructure.aishopping.vector.VectorStoreFactory;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

/**
 * Async AI shopping vector index builder: embeds all products into text/image
 * vectors and writes them to the vector store.
 */
@Service
public class AiShoppingIndexService {

    private static final Logger log = LoggerFactory.getLogger(AiShoppingIndexService.class);

    private final AiProviderClient providerClient;
    private final VectorStoreFactory vectorStoreFactory;
    private final ProductCatalogCache catalogCache;
    private final AiProviderConfig providerConfig;
    private final AiShoppingConfigRepository configRepository;
    private final AiShoppingFeatureGuard featureGuard;
    private final HttpClient httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .followRedirects(HttpClient.Redirect.NORMAL)
            .build();
    private final ExecutorService executor = Executors.newFixedThreadPool(4);
    private final IndexStatus status = new IndexStatus();

    public AiShoppingIndexService(AiProviderClient providerClient,
                                  VectorStoreFactory vectorStoreFactory,
                                  ProductCatalogCache catalogCache,
                                  AiProviderConfig providerConfig,
                                  AiShoppingConfigRepository configRepository,
                                  AiShoppingFeatureGuard featureGuard) {
        this.providerClient = providerClient;
        this.vectorStoreFactory = vectorStoreFactory;
        this.catalogCache = catalogCache;
        this.providerConfig = providerConfig;
        this.configRepository = configRepository;
        this.featureGuard = featureGuard;
    }

    public boolean isRunning() {
        return status.getState() == IndexStatus.State.RUNNING;
    }

    public IndexStatus getStatus() {
        return status;
    }

    public void rebuild(String type) {
        featureGuard.requireEnabled();
        if (isRunning()) {
            throw new IllegalStateException("index rebuild already in progress");
        }
        String effectiveType = type == null ? "all" : type.toLowerCase();
        startStatus(effectiveType);
        CompletableFuture.runAsync(() -> runBuild(effectiveType), executor);
    }

    private void startStatus(String type) {
        status.setState(IndexStatus.State.RUNNING);
        status.setCurrentType(type);
        status.setStartedAt(LocalDateTime.now());
        status.setLastError(null);
        status.setProcessed(0);
        status.setTotal(0);
    }

    private void runBuild(String type) {
        try {
            catalogCache.refresh();
            List<ProductDataProvider.ProductItem> products = catalogCache.all();
            status.setTotal(products.size());

            if ("text".equals(type) || "all".equals(type)) {
                rebuildText(products);
            }
            if ("image".equals(type) || "all".equals(type)) {
                rebuildImage(products);
            }

            markCompleted();
        } catch (Exception e) {
            markFailed(e);
        }
    }

    private void markCompleted() {
        status.setTextVectorCount((int) vectorStoreFactory.getStore().count(collectionText()));
        status.setImageVectorCount((int) vectorStoreFactory.getStore().count(collectionImage()));
        status.setState(IndexStatus.State.COMPLETED);
        status.setFinishedAt(LocalDateTime.now());
        persistIndexMeta("COMPLETED", LocalDateTime.now().toString());
    }

    private void markFailed(Exception e) {
        log.error("AI shopping index rebuild failed", e);
        status.setLastError(e.getMessage());
        status.setState(IndexStatus.State.FAILED);
        status.setFinishedAt(LocalDateTime.now());
    }

    private void persistIndexMeta(String state, String lastRebuilt) {
        configRepository.save(new AiShoppingConfig(
                AiProviderConfig.KEY_INDEX_STATUS, state, "index_status"));
        configRepository.save(new AiShoppingConfig(
                AiProviderConfig.KEY_INDEX_LAST_REBUILT, lastRebuilt, "last_rebuild_time"));
    }

    private void rebuildText(List<ProductDataProvider.ProductItem> products) {
        VectorStore store = vectorStoreFactory.getStore();
        String collection = collectionText();
        store.ensureCollection(collection, providerConfig.getSettings().getEmbeddingDim());

        AtomicInteger counter = new AtomicInteger();
        List<CompletableFuture<Void>> tasks = new ArrayList<>();
        for (ProductDataProvider.ProductItem product : products) {
            tasks.add(CompletableFuture.runAsync(() -> {
                try {
                    float[] vector = providerClient.embedText(textRepresentation(product));
                    VectorRecord record = new VectorRecord(product.id, product.id, vector, Collections.emptyMap());
                    store.upsert(collection, List.of(record));
                } catch (Exception e) {
                    log.warn("embed text failed for product {}: {}", product.id, e.getMessage());
                } finally {
                    int done = counter.incrementAndGet();
                    status.setProcessed(done);
                }
            }, executor));
        }
        CompletableFuture.allOf(tasks.toArray(new CompletableFuture[0])).join();
    }

    private void rebuildImage(List<ProductDataProvider.ProductItem> products) {
        VectorStore store = vectorStoreFactory.getStore();
        String collection = collectionImage();
        store.ensureCollection(collection, providerConfig.getSettings().getEmbeddingDim());

        AtomicInteger counter = new AtomicInteger();
        List<CompletableFuture<Void>> tasks = new ArrayList<>();
        for (ProductDataProvider.ProductItem product : products) {
            tasks.add(CompletableFuture.runAsync(() -> {
                try {
                    byte[] image = downloadImage(product.pic);
                    if (image == null || image.length == 0) {
                        return;
                    }
                    float[] vector = providerClient.embedImage(image);
                    Map<String, Object> metadata = Map.of("image_url", product.pic == null ? "" : product.pic);
                    VectorRecord record = new VectorRecord(product.id, product.id, vector, metadata);
                    store.upsert(collection, List.of(record));
                } catch (Exception e) {
                    log.warn("embed image failed for product {}: {}", product.id, e.getMessage());
                } finally {
                    int done = counter.incrementAndGet();
                    status.setProcessed(done);
                }
            }, executor));
        }
        CompletableFuture.allOf(tasks.toArray(new CompletableFuture[0])).join();
    }

    private byte[] downloadImage(String url) {
        if (url == null || url.isBlank()) {
            return new byte[0];
        }
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .timeout(Duration.ofSeconds(15))
                    .GET()
                    .build();
            HttpResponse<byte[]> response = httpClient.send(request, HttpResponse.BodyHandlers.ofByteArray());
            if (response.statusCode() >= 200 && response.statusCode() < 300) {
                return response.body();
            }
        } catch (Exception e) {
            log.debug("download image failed for {}: {}", url, e.getMessage());
        }
        return new byte[0];
    }

    private String textRepresentation(ProductDataProvider.ProductItem product) {
        StringBuilder sb = new StringBuilder();
        if (product.name != null) {
            sb.append(product.name).append(' ');
        }
        if (product.subTitle != null) {
            sb.append(product.subTitle).append(' ');
        }
        if (product.sku != null) {
            sb.append(product.sku).append(' ');
        }
        if (product.description != null) {
            sb.append(product.description).append(' ');
        }
        if (product.categoryId > 0) {
            sb.append("category ").append(product.categoryId).append(' ');
        }
        return sb.toString().trim();
    }

    private String collectionText() {
        return providerConfig.getSettings().getMilvusCollectionText();
    }

    private String collectionImage() {
        return providerConfig.getSettings().getMilvusCollectionImage();
    }
}
