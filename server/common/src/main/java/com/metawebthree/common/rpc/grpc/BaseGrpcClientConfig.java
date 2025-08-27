package com.metawebthree.common.rpc.grpc;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;

import java.util.concurrent.TimeUnit;

/**
 * Base gRPC Client Configuration
 * Provides common gRPC client configuration and channel management
 * Microservices can extend this class to add specific service client configurations
 */
@Slf4j
public abstract class BaseGrpcClientConfig {

    @Value("${grpc.client.keep-alive-time:30}")
    protected int keepAliveTime;

    @Value("${grpc.client.default.keep-alive-timeout:5}")
    protected int keepAliveTimeout;

    @Value("${grpc.client.max-inbound-message-size:10485760}")
    protected int maxInboundMessageSize;

    @Value("${grpc.client.idle-timeout:60}")
    protected int idleTimeout;

    /**
     * Create a managed channel with common configuration
     *
     * @param host Service host
     * @param port Service port
     * @return Configured ManagedChannel
     */
    protected ManagedChannel createChannel(String host, int port) {
        return ManagedChannelBuilder.forAddress(host, port)
                .usePlaintext() // Use TLS in production
                .keepAliveTime(keepAliveTime, TimeUnit.SECONDS)
                .keepAliveTimeout(keepAliveTimeout, TimeUnit.SECONDS)
                .maxInboundMessageSize(maxInboundMessageSize)
                .idleTimeout(idleTimeout, TimeUnit.SECONDS)
                .build();
    }

    /**
     * Create a managed channel with custom configuration
     *
     * @param host Service host
     * @param port Service port
     * @param customKeepAliveTime Custom keep alive time
     * @param customKeepAliveTimeout Custom keep alive timeout
     * @return Configured ManagedChannel
     */
    protected ManagedChannel createChannel(String host, int port, int customKeepAliveTime, int customKeepAliveTimeout) {
        return ManagedChannelBuilder.forAddress(host, port)
                .usePlaintext() // Use TLS in production
                .keepAliveTime(customKeepAliveTime, TimeUnit.SECONDS)
                .keepAliveTimeout(customKeepAliveTimeout, TimeUnit.SECONDS)
                .maxInboundMessageSize(maxInboundMessageSize)
                .idleTimeout(idleTimeout, TimeUnit.SECONDS)
                .build();
    }

    /**
     * Safely shutdown a channel
     *
     * @param channel Channel to shutdown
     * @param timeout Timeout in seconds
     */
    protected void shutdownChannel(ManagedChannel channel, int timeout) {
        if (channel != null && !channel.isShutdown()) {
            try {
                log.info("Shutting down gRPC channel to {}:{}", 
                    channel.authority() != null ? channel.authority() : "unknown", 
                    channel.getState(false));
                
                channel.shutdown().awaitTermination(timeout, TimeUnit.SECONDS);
                log.info("gRPC channel shutdown completed");
            } catch (InterruptedException e) {
                log.warn("gRPC channel shutdown interrupted", e);
                Thread.currentThread().interrupt();
            }
        }
    }

    /**
     * Get default keep alive time
     */
    protected int getDefaultKeepAliveTime() {
        return keepAliveTime;
    }

    /**
     * Get default keep alive timeout
     */
    protected int getDefaultKeepAliveTimeout() {
        return keepAliveTimeout;
    }

    /**
     * Get max inbound message size
     */
    protected int getMaxInboundMessageSize() {
        return maxInboundMessageSize;
    }

    /**
     * Get idle timeout
     */
    protected int getIdleTimeout() {
        return idleTimeout;
    }
}
