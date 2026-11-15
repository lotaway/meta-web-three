package com.metawebthree.digitaltwin.interfaces.websocket;

import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.common.utils.InternalTokenUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.server.ServerHttpRequest;
import org.springframework.http.server.ServerHttpResponse;
import org.springframework.web.socket.WebSocketHandler;
import org.springframework.web.socket.server.HandshakeInterceptor;

import java.util.Map;

/**
 * The gateway validates the user JWT and injects X-Internal-Token together
 * with the user headers. Only a caller that came through the gateway can
 * present a valid internal token, so the user identity is trusted here.
 */
public class DigitalTwinAuthHandshakeInterceptor implements HandshakeInterceptor {

    private static final Logger logger = LoggerFactory.getLogger(DigitalTwinAuthHandshakeInterceptor.class);
    private static final String INTERNAL_TOKEN_HEADER = "X-Internal-Token";

    private final InternalTokenUtil internalTokenUtil;

    public DigitalTwinAuthHandshakeInterceptor(InternalTokenUtil internalTokenUtil) {
        this.internalTokenUtil = internalTokenUtil;
    }

    @Override
    public boolean beforeHandshake(ServerHttpRequest request, ServerHttpResponse response,
                                   WebSocketHandler wsHandler, Map<String, Object> attributes) {
        String internalToken = request.getHeaders().getFirst(INTERNAL_TOKEN_HEADER);
        String userId = request.getHeaders().getFirst(HeaderConstants.USER_ID);
        String userRole = request.getHeaders().getFirst(HeaderConstants.USER_ROLE);

        if (internalToken == null || !internalTokenUtil.validate(internalToken)) {
            logger.warn("WebSocket handshake rejected: invalid or missing X-Internal-Token (gateway required)");
            response.setStatusCode(HttpStatus.UNAUTHORIZED);
            return false;
        }

        if (userId == null || userId.isBlank()) {
            logger.warn("WebSocket handshake rejected: gateway did not inject X-User-Id");
            response.setStatusCode(HttpStatus.UNAUTHORIZED);
            return false;
        }

        attributes.put("userId", userId);
        attributes.put("userRole", userRole);
        logger.info("WebSocket handshake authenticated by gateway: userId={}, role={}", userId, userRole);
        return true;
    }

    @Override
    public void afterHandshake(ServerHttpRequest request, ServerHttpResponse response,
                               WebSocketHandler wsHandler, Exception exception) {
    }
}
