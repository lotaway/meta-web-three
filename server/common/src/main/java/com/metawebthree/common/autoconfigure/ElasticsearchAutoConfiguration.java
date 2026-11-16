package com.metawebthree.common.autoconfigure;

import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Import;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import com.metawebthree.common.elasticsearch.ElasticsearchConfig;

/**
 * Elasticsearch client, only for services that declare the capability.
 */
@AutoConfiguration
@ConditionalOnClass(ElasticsearchClient.class)
@ConditionalOnProperty(prefix = "meta.common.storage.elasticsearch", name = "enabled", havingValue = "true")
@EnableConfigurationProperties(CommonStorageProperties.class)
@Import(ElasticsearchConfig.class)
public class ElasticsearchAutoConfiguration {
}
