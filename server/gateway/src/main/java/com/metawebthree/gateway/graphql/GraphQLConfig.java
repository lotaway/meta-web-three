package com.metawebthree.gateway.graphql;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import graphql.GraphQL;
import graphql.schema.GraphQLSchema;
import graphql.schema.idl.RuntimeWiring;
import graphql.schema.idl.SchemaGenerator;
import graphql.schema.idl.SchemaParser;
import graphql.schema.idl.TypeDefinitionRegistry;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.ResourcePatternResolver;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerCodecConfigurer;
import org.springframework.web.reactive.config.EnableWebFlux;
import org.springframework.web.reactive.config.WebFluxConfigurer;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

@Configuration
@EnableWebFlux
public class GraphQLConfig implements WebFluxConfigurer {

    @Value("classpath:graphql/*.graphqls")
    private Resource[] schemaResources;

    private final GraphQLDataProvider dataProvider;

    public GraphQLConfig(GraphQLDataProvider dataProvider) {
        this.dataProvider = dataProvider;
    }

    @Override
    public void configureHttpMessageCodecs(ServerCodecConfigurer configurer) {
        configurer.defaultCodecs().maxInMemorySize(16 * 1024 * 1024);
    }

    @Bean
    public GraphQLSchema graphQLSchema() throws IOException {
        StringBuilder schemaString = new StringBuilder();
        for (Resource resource : schemaResources) {
            schemaString.append(new String(resource.getInputStream().readAllBytes(), StandardCharsets.UTF_8));
        }

        TypeDefinitionRegistry typeRegistry = new SchemaParser().parse(schemaString.toString());
        RuntimeWiring runtimeWiring = buildRuntimeWiring();

        SchemaGenerator schemaGenerator = new SchemaGenerator();
        return schemaGenerator.makeExecutableSchema(typeRegistry, runtimeWiring);
    }

    private RuntimeWiring buildRuntimeWiring() {
        return RuntimeWiring.newRuntimeWiring()
                .type("Query", wiring -> wiring
                        .dataFetcher("product", dataProvider::getProduct)
                        .dataFetcher("products", dataProvider::getProducts)
                        .dataFetcher("productBySku", dataProvider::getProductBySku)
                        .dataFetcher("order", dataProvider::getOrder)
                        .dataFetcher("orders", dataProvider::getOrders)
                        .dataFetcher("orderByOrderNo", dataProvider::getOrderByOrderNo)
                        .dataFetcher("user", dataProvider::getUser)
                        .dataFetcher("users", dataProvider::getUsers)
                        .dataFetcher("category", dataProvider::getCategory)
                        .dataFetcher("categories", dataProvider::getCategories)
                        .dataFetcher("inventory", dataProvider::getInventory)
                        .dataFetcher("inventoryAlerts", dataProvider::getInventoryAlerts)
                )
                .type("Mutation", wiring -> wiring
                        .dataFetcher("createOrder", dataProvider::createOrder)
                        .dataFetcher("cancelOrder", dataProvider::cancelOrder)
                        .dataFetcher("payOrder", dataProvider::payOrder)
                        .dataFetcher("createProduct", dataProvider::createProduct)
                        .dataFetcher("updateProduct", dataProvider::updateProduct)
                        .dataFetcher("deleteProduct", dataProvider::deleteProduct)
                        .dataFetcher("addToCart", dataProvider::addToCart)
                        .dataFetcher("removeFromCart", dataProvider::removeFromCart)
                        .dataFetcher("clearCart", dataProvider::clearCart)
                )
                .build();
    }

    @Bean
    public GraphQL graphQL(GraphQLSchema graphQLSchema) {
        return GraphQL.newGraphQL(graphQLSchema).build();
    }

    @Bean
    public ObjectMapper objectMapper() {
        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        return mapper;
    }
}