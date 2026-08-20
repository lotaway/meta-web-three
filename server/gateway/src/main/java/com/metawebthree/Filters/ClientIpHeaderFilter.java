package com.metawebthree.Filters;

import com.metawebthree.common.constants.HeaderConstants;

import org.springframework.cloud.gateway.filter.GatewayFilterChain;
import org.springframework.cloud.gateway.filter.GlobalFilter;
import org.springframework.core.Ordered;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;

import java.net.InetSocketAddress;

import reactor.core.publisher.Mono;

import lombok.extern.slf4j.Slf4j;

@Component
@Slf4j
public class ClientIpHeaderFilter implements GlobalFilter, Ordered {

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        InetSocketAddress remoteAddress = exchange.getRequest().getRemoteAddress();
        if (remoteAddress == null || remoteAddress.getAddress() == null) {
            log.warn("Client IP unavailable for {}", exchange.getRequest().getPath());
            return chain.filter(exchange);
        }
        String clientIp = remoteAddress.getAddress().getHostAddress();
        ServerHttpRequest mutatedRequest = exchange.getRequest().mutate()
                .header(HeaderConstants.CLIENT_IP, clientIp)
                .build();
        return chain.filter(exchange.mutate().request(mutatedRequest).build());
    }

    @Override
    public int getOrder() {
        return -110;
    }
}