package com.metawebthree.cart.interfaces.graphql;

import com.apollographql.federation.graphqljava.Federation;
import com.metawebthree.cart.application.CartService;
import com.metawebthree.cart.dto.CartItemDTO;
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
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Configuration
public class CartSubgraphConfig {

    private final CartService cartService;
    private final ResourcePatternResolver resourceResolver;

    public CartSubgraphConfig(CartService cartService,
                               ResourcePatternResolver resourceResolver) {
        this.cartService = cartService;
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
                        .dataFetcher("cart", this::cartDataFetcher)
                )
                .type("Mutation", wiring -> wiring
                        .dataFetcher("addToCart", this::addToCartDataFetcher)
                        .dataFetcher("removeFromCart", this::removeFromCartDataFetcher)
                        .dataFetcher("clearCart", this::clearCartDataFetcher)
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
        if ("Cart".equals(typeName)) {
            Long id = Long.valueOf((String) representation.get("id"));
            var items = cartService.list(id);
            return buildData(id, items);
        }
        if ("CartItem".equals(typeName)) {
            String idStr = (String) representation.get("id");
            Map<String, Object> m = new ConcurrentHashMap<>();
            m.put("__typename", "CartItem");
            m.put("id", idStr);
            return m;
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

    private Map<String, Object> cartDataFetcher(DataFetchingEnvironment env) {
        String userId = env.getArgument("userId");
        Long memberId = Long.valueOf(userId);
        List<CartItemDTO> items = cartService.list(memberId);
        return buildCartResponse(memberId, items);
    }

    private Map<String, Object> addToCartDataFetcher(DataFetchingEnvironment env) {
        String productId = env.getArgument("productId");
        Integer quantity = env.getArgument("quantity");
        Long userId = getUserId(env);
        CartItemDTO dto = new CartItemDTO();
        dto.setMemberId(userId);
        dto.setProductId(Long.valueOf(productId));
        dto.setQuantity(quantity);
        dto.setPrice(BigDecimal.ZERO);
        cartService.add(dto);
        List<CartItemDTO> items = cartService.list(userId);
        return buildCartResponse(userId, items);
    }

    private Map<String, Object> removeFromCartDataFetcher(DataFetchingEnvironment env) {
        String cartItemId = env.getArgument("cartItemId");
        Long userId = getUserId(env);
        cartService.delete(userId, List.of(Long.valueOf(cartItemId)));
        List<CartItemDTO> items = cartService.list(userId);
        return buildCartResponse(userId, items);
    }

    private Map<String, Object> clearCartDataFetcher(DataFetchingEnvironment env) {
        Long userId = getUserId(env);
        cartService.clear(userId);
        return buildCartResponse(userId, List.of());
    }

    private Long getUserId(DataFetchingEnvironment env) {
        String userId = env.getArgument("userId");
        if (userId == null) {
            Map<String, Object> headers = env.getGraphQlContext().get("headers");
            if (headers != null && headers.containsKey("X-User-Id")) {
                return Long.valueOf((String) headers.get("X-User-Id"));
            }
            return 0L;
        }
        return Long.valueOf(userId);
    }

    private Map<String, Object> buildCartResponse(Long userId, List<CartItemDTO> items) {
        Map<String, Object> result = new ConcurrentHashMap<>();
        result.put("__typename", "Cart");
        result.put("id", String.valueOf(userId));
        result.put("userId", String.valueOf(userId));
        result.put("items", items.stream().map(this::toCartItemMap).toList());
        result.put("totalAmount", items.stream()
                .map(i -> i.getPrice().multiply(BigDecimal.valueOf(i.getQuantity())))
                .reduce(BigDecimal.ZERO, BigDecimal::add)
                .doubleValue());
        result.put("itemCount", items.size());
        return result;
    }

    private Map<String, Object> buildData(Long userId, List<CartItemDTO> items) {
        Map<String, Object> result = new ConcurrentHashMap<>();
        result.put("__typename", "Cart");
        result.put("id", String.valueOf(userId));
        result.put("userId", String.valueOf(userId));
        result.put("items", items.stream().map(this::toCartItemMap).toList());
        result.put("totalAmount", items.stream()
                .map(i -> i.getPrice().multiply(BigDecimal.valueOf(i.getQuantity())))
                .reduce(BigDecimal.ZERO, BigDecimal::add)
                .doubleValue());
        result.put("itemCount", items.size());
        return result;
    }

    private Map<String, Object> toCartItemMap(CartItemDTO item) {
        Map<String, Object> m = new ConcurrentHashMap<>();
        m.put("__typename", "CartItem");
        m.put("id", String.valueOf(item.getId()));
        m.put("productId", String.valueOf(item.getProductId()));
        m.put("quantity", item.getQuantity());
        m.put("price", item.getPrice() != null ? item.getPrice().doubleValue() : 0.0);
        BigDecimal subtotal = item.getPrice() != null
                ? item.getPrice().multiply(BigDecimal.valueOf(item.getQuantity()))
                : BigDecimal.ZERO;
        m.put("subtotal", subtotal.doubleValue());
        return m;
    }
}
