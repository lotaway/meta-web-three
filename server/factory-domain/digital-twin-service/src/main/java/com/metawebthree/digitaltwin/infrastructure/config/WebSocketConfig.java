package com.metawebthree.digitaltwin.infrastructure.config;

import com.metawebthree.digitaltwin.interfaces.websocket.DigitalTwinWebSocketHandler;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;

@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {

    private final DigitalTwinWebSocketHandler webSocketHandler;

    public WebSocketConfig(DigitalTwinWebSocketHandler webSocketHandler) {
        this.webSocketHandler = webSocketHandler;
    }

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(webSocketHandler, "/ws/digital-twin")
            .setAllowedOrigins("*");
    }
}