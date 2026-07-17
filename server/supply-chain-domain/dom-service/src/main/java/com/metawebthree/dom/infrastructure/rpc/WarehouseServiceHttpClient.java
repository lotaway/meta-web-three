package com.metawebthree.dom.infrastructure.rpc;

import com.metawebthree.dom.domain.service.WarehouseInfo;
import com.metawebthree.dom.domain.service.WarehouseServiceClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.List;

@Service
@ConditionalOnProperty(name = "dom.rpc.real-clients-enabled", havingValue = "true")
public class WarehouseServiceHttpClient implements WarehouseServiceClient {

    private static final Logger log = LoggerFactory.getLogger(WarehouseServiceHttpClient.class);
    private final RestTemplate restTemplate;
    private final String baseUrl;

    public WarehouseServiceHttpClient(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
        this.baseUrl = "http://warehouse-service/api/warehouse";
    }

    @Override
    public WarehouseInfo getWarehouse(Long warehouseId) {
        try {
            var response = restTemplate.getForEntity(
                    baseUrl + "/warehouses/{id}",
                    WarehouseInfoResponse.class, warehouseId);
            WarehouseInfoResponse body = response.getBody();
            if (body == null) return null;
            WarehouseInfo info = new WarehouseInfo();
            info.setId(body.getId());
            info.setName(body.getName());
            info.setRegion(body.getRegion());
            if (body.getLatitude() != null) info.setLatitude(body.getLatitude());
            if (body.getLongitude() != null) info.setLongitude(body.getLongitude());
            return info;
        } catch (Exception e) {
            log.warn("Failed to fetch warehouse {} from remote service: {}", warehouseId, e.getMessage());
            return null;
        }
    }

    @Override
    public Double getWarehouseDistance(String fromRegion, Long warehouseId) {
        WarehouseInfo warehouse = getWarehouse(warehouseId);
        if (warehouse == null) return 9999.0;
        if (fromRegion == null) return 500.0;
        if (fromRegion.equals(warehouse.getRegion())) return 50.0;
        return 200.0;
    }

    private static class WarehouseInfoResponse {
        private Long id;
        private String name;
        private String region;
        private Double latitude;
        private Double longitude;
        public Long getId() { return id; }
        public void setId(Long id) { this.id = id; }
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public String getRegion() { return region; }
        public void setRegion(String region) { this.region = region; }
        public Double getLatitude() { return latitude; }
        public void setLatitude(Double latitude) { this.latitude = latitude; }
        public Double getLongitude() { return longitude; }
        public void setLongitude(Double longitude) { this.longitude = longitude; }
    }
}
