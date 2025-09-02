package com.metawebthree.Config;

import java.util.List;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

// @TODO integrate all micro service api into one ui
@Configuration
public class SwaggerConfig {

    // @Bean
    // public List<SwaggerResource> swaggerResources(
    //         SwaggerResourcesProvider defaultResourcesProvider,
    //         RouteLocator routeLocator) {

    //     List<SwaggerResource> resources = new ArrayList<>();

    //     // Auto configure Swagger resources for each route
    //     routeLocator.getRoutes().filter(route -> route.getUri().getHost() != null)
    //             .filter(route -> !"api-gateway".equals(route.getUri().getHost()))
    //             .subscribe(route -> resources.add(createResource(
    //                     route.getId(),
    //                     "/" + route.getId() + "/v3/api-docs")));

    //     return resources;
    // }

    // private SwaggerResource createResource(String name, String location) {
    //     SwaggerResource resource = new SwaggerResource();
    //     resource.setName(name);
    //     resource.setLocation(location);
    //     resource.setSwaggerVersion("3.0");
    //     return resource;
    // }
}