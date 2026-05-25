package com.metawebthree.digitaltwin.infrastructure.config;

import com.metawebthree.digitaltwin.interfaces.websocket.DigitalTwinAuthHandshakeInterceptor;
import com.metawebthree.digitaltwin.interfaces.websocket.DigitalTwinWebSocketHandler;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;

import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {

    private final DigitalTwinWebSocketHandler webSocketHandler;

    @Value("${digital-twin.websocket.allowed-origins:}")
    private String allowedOrigins;

    public WebSocketConfig(DigitalTwinWebSocketHandler webSocketHandler) {
        this.webSocketHandler = webSocketHandler;
    }

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        var registration = registry.addHandler(webSocketHandler, "/ws/digital-twin")
            .addInterceptors(new DigitalTwinAuthHandshakeInterceptor());
        
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