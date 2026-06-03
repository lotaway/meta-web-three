package com.metawebthree.common.governance;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Configuration;

/**
 * Nacos Config Integration for Service Governance
 * 
 * This configuration is activated when Nacos config is enabled.
 * It allows dynamic refresh of service governance properties.
 * 
 * Enable Nacos config in application.yml:
 * <pre>
 * spring:
 *   cloud:
 *     nacos:
 *       config:
 *         server-addr: ${NACOS_SERVER:localhost:8848}
 *         namespace: ${NACOS_NAMESPACE:}
 *         group: DEFAULT_GROUP
 *         file-extension: yml
 *         shared-configs:
 *           - data-id: governance-config.yml
 *             group: DEFAULT_GROUP
 *             refresh: true
 * </pre>
 */
@Configuration
@EnableConfigurationProperties(ServiceGovernanceProperties.class)
@ConditionalOnProperty(name = "spring.cloud.nacos.config.enabled", havingValue = "true", matchIfMissing = false)
@Slf4j
public class NacosGovernanceConfigIntegration {

    // This class is automatically activated when Nacos config is enabled
    // The ServiceGovernanceProperties will be refreshed automatically when config changes

    public NacosGovernanceConfigIntegration() {
        log.info("Nacos governance config integration enabled");
    }
}
