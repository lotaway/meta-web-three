package com.metawebthree.digitaltwin.interfaces.websocket;

import com.metawebthree.common.constants.HeaderConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.server.ServerHttpRequest;
import org.springframework.http.server.ServerHttpResponse;
import org.springframework.web.socket.WebSocketHandler;
import org.springframework.web.socket.server.HandshakeInterceptor;

import java.util.Map;

public class DigitalTwinAuthHandshakeInterceptor implements HandshakeInterceptor {

    private static final Logger logger = LoggerFactory.getLogger(DigitalTwinAuthHandshakeInterceptor.class);

    @Override
    public boolean beforeHandshake(ServerHttpRequest request, ServerHttpResponse response,
                                   WebSocketHandler wsHandler, Map<String, Object> attributes) {
        String userId = request.getHeaders().getFirst(HeaderConstants.USER_ID);
        String userRole = request.getHeaders().getFirst(HeaderConstants.USER_ROLE);
        String token = request.getHeaders().getFirst("Authorization");

        if (userId == null || token == null) {
            logger.warn("WebSocket handshake rejected: missing authentication headers");
            response.setStatusCode(HttpStatus.UNAUTHORIZED);
            return false;
        }

        attributes.put("userId", userId);
        attributes.put("userRole", userRole);
        logger.info("WebSocket handshake authenticated: userId={}, role={}", userId, userRole);
        return true;
    }

    @Override
    public void afterHandshake(ServerHttpRequest request, ServerHttpResponse response,
                               WebSocketHandler wsHandler, Exception exception) {
    }
}
