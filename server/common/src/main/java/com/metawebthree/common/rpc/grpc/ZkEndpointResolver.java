package com.metawebthree.common.rpc.grpc;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.curator.framework.CuratorFramework;
import org.apache.curator.framework.CuratorFrameworkFactory;
import org.apache.curator.retry.ExponentialBackoffRetry;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;

@Slf4j
@Component
@RequiredArgsConstructor
public class ZkEndpointResolver {

    @Value("${spring.cloud.zookeeper.connect-string:localhost:2181}")
    private String zkConnectString;

    @Value("${grpc.discovery.path-template:/grpc/{serviceName}/providers}")
    private String pathTemplate;

    public record HostAndPort(String host, int port) {}

    public List<HostAndPort> resolve(String serviceName) {
        String path = pathTemplate.replace("{serviceName}", serviceName);
        try (CuratorFramework cf = CuratorFrameworkFactory.newClient(
                zkConnectString, new ExponentialBackoffRetry(1000, 3))) {
            cf.start();
            if (cf.checkExists().forPath(path) == null) {
                log.warn("ZK path not found: {}", path);
                return List.of();
            }
            List<String> children = cf.getChildren().forPath(path);
            return children.stream().map(this::parseNode).filter(Objects::nonNull).toList();
        } catch (Exception e) {
            log.error("Failed to resolve endpoints for {}", serviceName, e);
            return List.of();
        }
    }

    private HostAndPort parseNode(String node) {
        String PRE_FIX = "grpc://";
        String value = node.startsWith(PRE_FIX) ? node.substring(PRE_FIX.length()) : node;
        String[] parts = value.split(":");
        if (parts.length != 2) return null;
        try {
            return new HostAndPort(parts[0], Integer.parseInt(parts[1]));
        } catch (NumberFormatException e) {
            return null;
        }
    }

    public HostAndPort pickOne(List<HostAndPort> endpoints) {
        if (endpoints == null || endpoints.isEmpty()) return null;
        return endpoints.get(ThreadLocalRandom.current().nextInt(endpoints.size()));
    }
}
