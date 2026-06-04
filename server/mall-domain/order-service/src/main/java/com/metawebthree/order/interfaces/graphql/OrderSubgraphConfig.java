package com.metawebthree.order.interfaces.graphql;

import com.apollographql.federation.graphqljava.Federation;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.order.application.OrderApplicationService;
import com.metawebthree.order.domain.model.OrderDO;
import com.metawebthree.order.domain.model.OrderItemDO;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderItemMapper;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderMapper;
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
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@Configuration
public class OrderSubgraphConfig {

    private final OrderMapper orderMapper;
    private final OrderItemMapper orderItemMapper;
    private final OrderApplicationService orderApplicationService;
    private final ResourcePatternResolver resourceResolver;

    public OrderSubgraphConfig(OrderMapper orderMapper, OrderItemMapper orderItemMapper,
                                OrderApplicationService orderApplicationService,
                                ResourcePatternResolver resourceResolver) {
        this.orderMapper = orderMapper;
        this.orderItemMapper = orderItemMapper;
        this.orderApplicationService = orderApplicationService;
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
                        .dataFetcher("order", this::orderDataFetcher)
                        .dataFetcher("orders", this::ordersDataFetcher)
                        .dataFetcher("orderByOrderNo", this::orderByOrderNoDataFetcher)
                )
                .type("Mutation", wiring -> wiring
                        .dataFetcher("createOrder", this::createOrderDataFetcher)
                        .dataFetcher("cancelOrder", this::cancelOrderDataFetcher)
                        .dataFetcher("payOrder", this::payOrderDataFetcher)
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
        if ("Order".equals(typeName)) {
            String id = (String) representation.get("id");
            OrderDO order = orderMapper.selectById(Long.valueOf(id));
            return order != null ? toOrderMap(order) : null;
        }
        if ("OrderItem".equals(typeName)) {
            String id = (String) representation.get("id");
            OrderItemDO item = orderItemMapper.selectById(Long.valueOf(id));
            return item != null ? toOrderItemMap(item) : null;
        }
        return null;
    }

    private GraphQLObjectType resolveEntityType(TypeResolutionEnvironment env) {
        Object src = env.getObject();
        if (src instanceof Map) {
            String tn = (String) ((Map<?, ?>) src).get("__typename");
            return env.getSchema().getObjectType(tn);
        }
        return null;
    }

    private Object orderDataFetcher(DataFetchingEnvironment env) {
        String id = env.getArgument("id");
        OrderDO order = orderMapper.selectById(Long.valueOf(id));
        return order != null ? toOrderMap(order) : null;
    }

    private Map<String, Object> ordersDataFetcher(DataFetchingEnvironment env) {
        Integer page = env.getArgument("page");
        Integer size = env.getArgument("size");
        if (page == null) page = 0;
        if (size == null) size = 10;
        long total = orderMapper.selectCount(null);
        List<OrderDO> orders = orderMapper.selectList(
                new LambdaQueryWrapper<OrderDO>()
                        .orderByDesc(OrderDO::getCreatedAt)
                        .last("limit " + size + " offset " + Math.max(0, page * size)));
        List<Map<String, Object>> edges = orders.stream().map(o -> {
            Map<String, Object> edge = new ConcurrentHashMap<>();
            edge.put("node", toOrderMap(o));
            return edge;
        }).toList();
        Map<String, Object> pageInfo = new ConcurrentHashMap<>();
        pageInfo.put("hasNextPage", (page + 1) * size < total);
        pageInfo.put("hasPreviousPage", page > 0);
        Map<String, Object> result = new ConcurrentHashMap<>();
        result.put("edges", edges);
        result.put("pageInfo", pageInfo);
        result.put("totalCount", total);
        return result;
    }

    private Object orderByOrderNoDataFetcher(DataFetchingEnvironment env) {
        String orderNo = env.getArgument("orderNo");
        OrderDO order = orderMapper.selectOne(
                new LambdaQueryWrapper<OrderDO>().eq(OrderDO::getOrderNo, orderNo));
        return order != null ? toOrderMap(order) : null;
    }

    private Map<String, Object> createOrderDataFetcher(DataFetchingEnvironment env) {
        Map<String, Object> input = env.getArgument("input");
        Long orderId = IdWorker.getId();
        String orderNo = String.valueOf(IdWorker.getId());
        BigDecimal total = BigDecimal.ZERO;
        List<Map<String, Object>> items = (List<Map<String, Object>>) input.get("items");
        if (items != null) {
            for (Map<String, Object> item : items) {
                total = total.add(BigDecimal.valueOf(((Number) item.get("price")).doubleValue())
                        .multiply(BigDecimal.valueOf(((Number) item.get("quantity")).intValue())));
            }
        }
        OrderDO order = OrderDO.builder()
                .id(orderId).orderNo(orderNo).orderStatus("CREATED")
                .orderType("NORMAL").orderAmount(total)
                .orderRemark((String) input.get("orderRemark"))
                .build();
        orderMapper.insert(order);
        Map<String, Object> result = toOrderMap(order);
        result.put("id", String.valueOf(orderId));
        result.put("orderNo", orderNo);
        return result;
    }

    private Map<String, Object> cancelOrderDataFetcher(DataFetchingEnvironment env) {
        String id = env.getArgument("id");
        OrderDO order = orderMapper.selectById(Long.valueOf(id));
        if (order != null) {
            order.setOrderStatus("CANCELED");
            orderMapper.updateById(order);
        }
        Map<String, Object> result = new ConcurrentHashMap<>();
        result.put("id", id);
        result.put("status", "CANCELED");
        return result;
    }

    private Map<String, Object> payOrderDataFetcher(DataFetchingEnvironment env) {
        String id = env.getArgument("id");
        orderApplicationService.paySuccess(Long.valueOf(id), 1);
        Map<String, Object> result = new ConcurrentHashMap<>();
        result.put("id", id);
        result.put("status", "PAID");
        return result;
    }

    private Map<String, Object> toOrderMap(OrderDO o) {
        Map<String, Object> m = new ConcurrentHashMap<>();
        m.put("id", String.valueOf(o.getId()));
        m.put("orderNo", o.getOrderNo());
        m.put("userId", String.valueOf(o.getUserId()));
        m.put("totalAmount", o.getOrderAmount() != null ? o.getOrderAmount().doubleValue() : 0.0);
        m.put("status", o.getOrderStatus());
        m.put("shippingAddress", "");
        m.put("paymentMethod", "");
        m.put("createdAt", o.getCreatedAt() != null ? o.getCreatedAt().toString() : null);
        m.put("updatedAt", o.getUpdatedAt() != null ? o.getUpdatedAt().toString() : null);
        List<OrderItemDO> items = orderItemMapper.selectList(
                new LambdaQueryWrapper<OrderItemDO>().eq(OrderItemDO::getOrderId, o.getId()));
        m.put("items", items.stream().map(this::toOrderItemMap).toList());
        return m;
    }

    private Map<String, Object> toOrderItemMap(OrderItemDO item) {
        Map<String, Object> m = new ConcurrentHashMap<>();
        m.put("id", String.valueOf(item.getId()));
        m.put("productId", String.valueOf(item.getProductId()));
        m.put("quantity", item.getQuantity());
        m.put("price", item.getUnitPrice() != null ? item.getUnitPrice().doubleValue() : 0.0);
        m.put("subtotal", item.getTotalPrice() != null ? item.getTotalPrice().doubleValue() : 0.0);
        return m;
    }
}
