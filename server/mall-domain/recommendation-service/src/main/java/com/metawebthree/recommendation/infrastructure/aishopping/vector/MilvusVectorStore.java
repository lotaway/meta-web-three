package com.metawebthree.recommendation.infrastructure.aishopping.vector;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Milvus vector store (RESTful API, port 19530). Collections use the COSINE
 * metric with AUTOINDEX.
 */
public class MilvusVectorStore implements VectorStore {

    private final String baseUrl;
    private final String apiToken;
    private final int timeoutMs;
    private final ObjectMapper objectMapper = new ObjectMapper();
    private final HttpClient httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(5))
            .build();

    public MilvusVectorStore(String host, int port, String apiToken) {
        this.baseUrl = "http://" + host + ":" + port;
        this.apiToken = apiToken == null ? "" : apiToken;
        this.timeoutMs = 15000;
    }

    @Override
    public String name() {
        return "milvus";
    }

    @Override
    public void ensureCollection(String collection, int dim) {
        if (collectionExists(collection)) {
            return;
        }
        ObjectNode body = objectMapper.createObjectNode();
        body.put("collectionName", collection);
        body.put("dimension", dim);
        body.put("metricType", "COSINE");
        body.put("primaryFieldName", "id");
        body.put("idType", "Int64");
        body.put("vectorFieldName", "vector");
        ArrayNode fields = body.putArray("fields");
        ObjectNode productId = fields.addObject();
        productId.put("fieldName", "product_id");
        productId.put("dataType", "Int64");
        ObjectNode imageUrl = fields.addObject();
        imageUrl.put("fieldName", "image_url");
        imageUrl.put("dataType", "VarChar");
        imageUrl.putObject("elementTypeParams").put("max_length", 1024);
        post("/v2/vectordb/collections/create", body);
    }

    @Override
    public void upsert(String collection, List<VectorRecord> records) {
        if (records.isEmpty()) {
            return;
        }
        ArrayNode data = objectMapper.createArrayNode();
        for (VectorRecord record : records) {
            ObjectNode row = data.addObject();
            row.put("id", record.getId());
            row.put("product_id", record.getProductId());
            ArrayNode vector = row.putArray("vector");
            for (float v : record.getVector()) {
                vector.add(v);
            }
            Object imageUrl = record.getMetadata() != null ? record.getMetadata().get("image_url") : null;
            if (imageUrl != null) {
                row.put("image_url", String.valueOf(imageUrl));
            }
        }
        ObjectNode body = objectMapper.createObjectNode();
        body.put("collectionName", collection);
        body.set("data", data);
        post("/v2/vectordb/entities/insert", body);
    }

    @Override
    public List<VectorHit> search(String collection, float[] query, int topK) {
        ArrayNode data = objectMapper.createArrayNode();
        ArrayNode vector = data.addArray();
        for (float v : query) {
            vector.add(v);
        }
        ObjectNode body = objectMapper.createObjectNode();
        body.put("collectionName", collection);
        body.set("data", data);
        body.put("annsField", "vector");
        body.put("limit", topK);
        ArrayNode outputFields = body.putArray("outputFields");
        outputFields.add("product_id");

        JsonNode response = post("/v2/vectordb/entities/search", body);
        return parseHits(response);
    }

    private List<VectorHit> parseHits(JsonNode response) {
        List<VectorHit> hits = new ArrayList<>();
        JsonNode dataNode = response.path("data");
        if (dataNode.isArray()) {
            dataNode.forEach(hit -> {
                long id = hit.path("id").asLong();
                float score = (float) hit.path("distance").asDouble();
                long productId = hit.path("entity").path("product_id").asLong();
                Map<String, Object> metadata = new LinkedHashMap<>();
                metadata.put("product_id", productId);
                hits.add(new VectorHit(id, productId, score, metadata));
            });
        }
        return hits;
    }

    @Override
    public long count(String collection) {
        ObjectNode body = objectMapper.createObjectNode();
        body.put("collectionName", collection);
        ArrayNode outputFields = body.putArray("outputFields");
        outputFields.add("count(*)");
        try {
            JsonNode response = post("/v2/vectordb/entities/query", body);
            JsonNode data = response.path("data");
            if (data.isArray() && !data.isEmpty()) {
                return data.get(0).path("count(*)").asLong();
            }
        } catch (IllegalStateException ignored) {
            // returns 0 when the collection is missing or unreachable
        }
        return 0L;
    }

    @Override
    public void drop(String collection) {
        ObjectNode body = objectMapper.createObjectNode();
        body.put("collectionName", collection);
        post("/v2/vectordb/collections/drop", body);
    }

    private boolean collectionExists(String collection) {
        ObjectNode body = objectMapper.createObjectNode();
        body.put("collectionName", collection);
        try {
            JsonNode response = post("/v2/vectordb/collections/describe", body);
            return response.path("code").asInt() == 0;
        } catch (Exception e) {
            return false;
        }
    }

    private JsonNode post(String path, JsonNode body) {
        try {
            return sendRequest(path, body);
        } catch (IllegalStateException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException("milvus request failed: " + e.getMessage(), e);
        }
    }

    private JsonNode sendRequest(String path, JsonNode body) throws Exception {
        HttpRequest.Builder builder = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .timeout(Duration.ofMillis(timeoutMs))
                .header("Content-Type", "application/json");
        if (apiToken != null && !apiToken.isBlank()) {
            builder.header("Authorization", "Bearer " + apiToken);
        }
        builder.POST(HttpRequest.BodyPublishers.ofString(objectMapper.writeValueAsString(body)));
        HttpResponse<String> response = httpClient.send(builder.build(),
                HttpResponse.BodyHandlers.ofString());
        JsonNode root = objectMapper.readTree(response.body());
        if (response.statusCode() >= 200 && response.statusCode() < 300
                && (root.path("code").isMissingNode() || root.path("code").asInt() == 0)) {
            return root;
        }
        throw new IllegalStateException("milvus HTTP " + response.statusCode() + ": " + response.body());
    }
}
