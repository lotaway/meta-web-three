package com.metawebthree.common.autoconfigure;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Feature switches for shared storage capabilities.
 *
 * <p>Every capability is disabled by default. A service opts in from its own
 * {@code application.yml}, for example:</p>
 *
 * <pre>
 * meta:
 *   common:
 *     storage:
 *       database:
 *         enabled: true
 *       redis:
 *         enabled: true
 * </pre>
 *
 * <p>Connection and pool settings stay under their existing Spring namespaces
 * ({@code spring.datasource}, {@code spring.data.redis}, {@code elasticsearch.hosts},
 * {@code rocketmq.client}) so each service can still override them independently.</p>
 */
@Data
@ConfigurationProperties(prefix = "meta.common.storage")
public class CommonStorageProperties {

    private final Capability database = new Capability();
    private final Capability redis = new Capability();
    private final Capability elasticsearch = new Capability();
    private final Capability rocketmq = new Capability();
    private final Capability kafka = new Capability();
    private final Capability object = new Capability();

    @Data
    public static class Capability {
        private boolean enabled;
    }
}
