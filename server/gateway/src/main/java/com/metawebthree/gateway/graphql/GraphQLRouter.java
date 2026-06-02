package com.metawebthree.gateway.graphql;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.RouterFunctions;
import org.springframework.web.reactive.function.server.ServerResponse;

import static org.springframework.web.reactive.function.server.RequestPredicates.POST;
import static org.springframework.web.reactive.function.server.RequestPredicates.GET;
import static org.springframework.web.reactive.function.server.RequestPredicates.accept;

@Configuration
public class GraphQLRouter {

    @Bean
    public RouterFunction<ServerResponse> graphQLRoutes(GraphQLHandler graphQLHandler) {
        return RouterFunctions
                .route(POST("/graphql").and(accept(MediaType.APPLICATION_JSON)), graphQLHandler::handleGraphQL)
                .andRoute(GET("/graphql").and(accept(MediaType.APPLICATION_JSON)), graphQLHandler::handleGraphQL);
    }
}