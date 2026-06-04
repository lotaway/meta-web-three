package com.metawebthree.inventory.interfaces.graphql;

import com.apollographql.federation.graphqljava.Federation;
import com.metawebthree.inventory.application.InventoryAlertAppService;
import com.metawebthree.inventory.application.InventoryApplicationService;
import com.metawebthree.inventory.domain.entity.alert.InventoryAlert;
import com.metawebthree.inventory.domain.repository.alert.InventoryAlertRepository;
import graphql.GraphQL;
import graphql.schema.GraphQLSchema;
import graphql.schema.idl.*;
import graphql.schema.DataFetchingEnvironment;
import graphql.TypeResolutionEnvironment;
import graphql.schema.GraphQLObjectType;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.ResourcePatternResolver;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Configuration
public class InventorySubgraphConfig {

    private final InventoryApplicationService inventoryService;
    private final InventoryAlertAppService inventoryAlertAppService;
    private final InventoryAlertRepository inventoryAlertRepository;
    private final ResourcePatternResolver resourceResolver;

    public InventorySubgraphConfig(InventoryApplicationService inventoryService,
                                   InventoryAlertAppService inventoryAlertAppService,
                                   InventoryAlertRepository inventoryAlertRepository,
                                   ResourcePatternResolver resourceResolver) {
        this.inventoryService = inventoryService;
        this.inventoryAlertAppService = inventoryAlertAppService;
        this.inventoryAlertRepository = inventoryAlertRepository;
        this.resourceResolver = resourceResolver;
    }

    @Bean
    public GraphQL graphQL() throws IOException {
        TypeDefinitionRegistry typeRegistry = buildSchema();
        RuntimeWiring wiring = buildRuntimeWiring();
        GraphQLSchema schema = buildFederationSchema(typeRegistry, wiring);
        return GraphQL.newGraphQL(schema).build();
    }

    private TypeDefinitionRegistry buildSchema() throws IOException {
        Resource[] resources = resourceResolver.getResources("classpath:graphql/*.graphqls");
        StringBuilder sb = new StringBuilder();
        for (Resource r : resources) {
            sb.append(new String(r.getInputStream().readAllBytes(), StandardCharsets.UTF_8));
        }
        return new SchemaParser().parse(sb.toString());
    }

    private RuntimeWiring buildRuntimeWiring() {
        return RuntimeWiring.newRuntimeWiring()
                .type("Query", wiring -> wiring
                        .dataFetcher("inventory", this::inventoryDataFetcher)
                        .dataFetcher("inventoryAlerts", this::inventoryAlertsDataFetcher)
                )
                .build();
    }

    private GraphQLSchema buildFederationSchema(TypeDefinitionRegistry registry, RuntimeWiring wiring) {
        return Federation.transform(registry, wiring)
                .fetchEntities(this::fetchEntity)
                .resolveEntityType(this::resolveEntityType)
                .build();
    }

    private Object fetchEntity(DataFetchingEnvironment env) {
        List<Map<String, Object>> representations = env.getArgument("representations");
        Map<String, Object> representation = representations.get(0);
        String typeName = (String) representation.get("__typename");
        if ("Inventory".equals(typeName)) {
            Object idObj = representation.get("productId");
            String productId = idObj.toString();
            var inventoryList = inventoryService.queryBySkuCode(productId);
            if (inventoryList.isEmpty()) return null;
            var dto = inventoryList.get(0);
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("productId", productId);
            result.put("quantity", dto.getTotalQuantity());
            result.put("availableQuantity", dto.getAvailableQuantity());
            result.put("reservedQuantity", dto.getReservedQuantity());
            result.put("warehouseId", dto.getWarehouseId() != null ? dto.getWarehouseId().toString() : null);
            result.put("updatedAt", dto.getUpdatedAt() != null ? dto.getUpdatedAt().toString() : null);
            return result;
        }
        if ("InventoryAlert".equals(typeName)) {
            Object idObj = representation.get("id");
            Long alertId = idObj instanceof Number ? ((Number) idObj).longValue() : Long.valueOf(idObj.toString());
            InventoryAlert alert = inventoryAlertRepository.findById(alertId);
            if (alert == null) return null;
            return alertToMap(alert);
        }
        return null;
    }

    private GraphQLObjectType resolveEntityType(TypeResolutionEnvironment env) {
        Object src = env.getObject();
        if (src instanceof Map) {
            String typeName = (String) ((Map<?, ?>) src).get("__typename");
            return env.getSchema().getObjectType(typeName);
        }
        return null;
    }

    private Map<String, Object> inventoryDataFetcher(DataFetchingEnvironment env) {
        String productId = env.getArgument("productId");
        var inventoryList = inventoryService.queryBySkuCode(productId);
        if (inventoryList.isEmpty()) return null;
        var dto = inventoryList.get(0);
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("productId", productId);
        result.put("quantity", dto.getTotalQuantity());
        result.put("availableQuantity", dto.getAvailableQuantity());
        result.put("reservedQuantity", dto.getReservedQuantity());
        result.put("warehouseId", dto.getWarehouseId() != null ? dto.getWarehouseId().toString() : null);
        result.put("updatedAt", dto.getUpdatedAt() != null ? dto.getUpdatedAt().toString() : null);
        return result;
    }

    private List<Map<String, Object>> inventoryAlertsDataFetcher(DataFetchingEnvironment env) {
        String status = env.getArgument("status");
        List<InventoryAlert> alerts;
        if (status == null || status.isBlank()) {
            alerts = inventoryAlertAppService.getActiveAlerts();
        } else {
            alerts = inventoryAlertRepository.findAll().stream()
                    .filter(a -> a.getStatus().name().equalsIgnoreCase(status))
                    .collect(Collectors.toList());
        }
        return alerts.stream().map(this::alertToMap).toList();
    }

    private Map<String, Object> alertToMap(InventoryAlert alert) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("id", alert.getId().toString());
        result.put("productId", alert.getSkuCode());
        result.put("alertType", alert.getAlertType().name());
        result.put("currentStock", alert.getCurrentQuantity());
        result.put("threshold", alert.getThresholdValue());
        result.put("status", alert.getStatus().name());
        result.put("createdAt", alert.getCreatedAt() != null ? alert.getCreatedAt().toString() : null);
        result.put("resolvedAt", alert.getResolvedAt() != null ? alert.getResolvedAt().toString() : null);
        return result;
    }
}
