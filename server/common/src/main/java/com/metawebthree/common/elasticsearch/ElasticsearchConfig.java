package com.metawebthree.common.elasticsearch;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.json.jackson.JacksonJsonpMapper;
import co.elastic.clients.transport.ElasticsearchTransport;
import co.elastic.clients.transport.rest_client.RestClientTransport;
import org.apache.http.HttpHost;
import org.elasticsearch.client.RestClient;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class ElasticsearchConfig {

    @Value("${elasticsearch.hosts:localhost:9200}")
    private String hosts;

    @Bean
    public ElasticsearchClient elasticsearchClient() {
        HttpHost[] httpHosts = parseHosts(hosts);
        RestClient restClient = RestClient.builder(httpHosts).build();
        ElasticsearchTransport transport = new RestClientTransport(restClient, new JacksonJsonpMapper());
        return new ElasticsearchClient(transport);
    }

    private HttpHost[] parseHosts(String hosts) {
        String[] hostArray = hosts.split(",");
        HttpHost[] httpHosts = new HttpHost[hostArray.length];
        for (int i = 0; i < hostArray.length; i++) {
            String[] parts = hostArray[i].trim().split(":");
            httpHosts[i] = new HttpHost(parts[0], Integer.parseInt(parts[1]));
        }
        return httpHosts;
    }
}
