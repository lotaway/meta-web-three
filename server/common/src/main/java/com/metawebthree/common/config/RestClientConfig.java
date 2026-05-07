package com.metawebthree.common.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
public class RestClientConfig {

    @Bean
    public RestTemplate restTemplate() {
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(5000);
        factory.setReadTimeout(10000);
        return new RestTemplate(factory);
    }

    @Bean
    public WebClient webClient() {
        return WebClient.builder()
                .clientConnector(new org.springframework.http.client.reactive.ReactorClientHttpConnector(
                        reactor.netty.http.client.HttpClient.create()
                                .responseTimeout(java.time.Duration.ofSeconds(10))
                ))
                .defaultHeader(org.springframework.http.HttpHeaders.CONTENT_TYPE, org.springframework.http.MediaType.APPLICATION_JSON_VALUE)
                .build();
    }
}