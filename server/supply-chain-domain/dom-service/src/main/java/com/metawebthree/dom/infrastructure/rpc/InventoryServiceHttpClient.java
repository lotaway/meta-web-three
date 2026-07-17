package com.metawebthree.dom.infrastructure.rpc;

import com.metawebthree.dom.domain.service.InventoryServiceClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.Map;

@Service
@ConditionalOnProperty(name = "dom.rpc.real-clients-enabled", havingValue = "true")
public class InventoryServiceHttpClient implements InventoryServiceClient {

    private static final Logger log = LoggerFactory.getLogger(InventoryServiceHttpClient.class);
    private final RestTemplate restTemplate;
    private final String baseUrl;

    public InventoryServiceHttpClient(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
        this.baseUrl = "http://inventory-service/api/inventory";
    }

    @Override
    public Integer checkInventory(String skuCode, Long warehouseId) {
        try {
            var response = restTemplate.getForEntity(
                    baseUrl + "?skuCode={skuCode}&warehouseId={warehouseId}",
                    InventoryResponse.class, skuCode, warehouseId);
            InventoryResponse body = response.getBody();
            if (body == null) return 0;
            return body.getQuantity() != null ? body.getQuantity() : 0;
        } catch (Exception e) {
            log.warn("Failed to check inventory for {} at warehouse {}: {}", skuCode, warehouseId, e.getMessage());
            return 0;
        }
    }

    private static class InventoryResponse {
        private Long id;
        private String skuCode;
        private Long warehouseId;
        private Integer quantity;
        public Long getId() { return id; }
        public void setId(Long id) { this.id = id; }
        public String getSkuCode() { return skuCode; }
        public void setSkuCode(String skuCode) { this.skuCode = skuCode; }
        public Long getWarehouseId() { return warehouseId; }
        public void setWarehouseId(Long warehouseId) { this.warehouseId = warehouseId; }
        public Integer getQuantity() { return quantity; }
        public void setQuantity(Integer quantity) { this.quantity = quantity; }
    }
}
