package com.metawebthree.digitaltwin.infrastructure.config;

import com.metawebthree.common.utils.InternalTokenUtil;
import com.metawebthree.digitaltwin.interfaces.websocket.DigitalTwinAuthHandshakeInterceptor;
import com.metawebthree.digitaltwin.interfaces.websocket.DigitalTwinWebSocketHandler;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {

    private final DigitalTwinWebSocketHandler webSocketHandler;
    private final InternalTokenUtil internalTokenUtil;

    @Value("${digital-twin.websocket.allowed-origins:}")
    private String allowedOrigins;

    public WebSocketConfig(
            DigitalTwinWebSocketHandler webSocketHandler,
            @Value("${internal-token.secret:mwt-internal-secret-2024}") String internalTokenSecret,
            @Value("${internal-token.timeout-seconds:5}") long internalTokenTimeoutSeconds) {
        this.webSocketHandler = webSocketHandler;
        // FactoryServiceApplication restricts component scanning, so construct
        // the shared HMAC validator directly from the same config the gateway uses.
        this.internalTokenUtil = new InternalTokenUtil(internalTokenSecret, internalTokenTimeoutSeconds);
    }

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        var registration = registry.addHandler(webSocketHandler, "/ws/digital-twin")
            .addInterceptors(new DigitalTwinAuthHandshakeInterceptor(internalTokenUtil));

        if (allowedOrigins != null && !allowedOrigins.isEmpty()) {
            String[] origins = allowedOrigins.split(",");
            registration.setAllowedOrigins(origins);
            log.info("WebSocket allowed origins: {}", (Object[]) origins);
        } else {
            // Default: no cross-origin allowed in production
            registration.setAllowedOrigins();
            log.info("WebSocket: no cross-origin allowed (empty allowed-origins config)");
        }
    }
}
