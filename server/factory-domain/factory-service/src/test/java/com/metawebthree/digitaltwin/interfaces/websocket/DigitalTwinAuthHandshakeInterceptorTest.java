package com.metawebthree.digitaltwin.interfaces.websocket;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.http.server.ServletServerHttpRequest;
import org.springframework.http.server.ServletServerHttpResponse;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;
import org.springframework.web.socket.WebSocketHandler;

import com.metawebthree.common.utils.InternalTokenUtil;

class DigitalTwinAuthHandshakeInterceptorTest {

    private final InternalTokenUtil internalTokenUtil = new InternalTokenUtil("shared-secret", 5);
    private final DigitalTwinAuthHandshakeInterceptor interceptor =
            new DigitalTwinAuthHandshakeInterceptor(internalTokenUtil);

    @Test
    void acceptsHandshakeWithGatewayTokenAndUserId() {
        MockHttpServletRequest servletRequest = request(internalTokenUtil.generate(), "7", "ADMIN");
        MockHttpServletResponse servletResponse = new MockHttpServletResponse();
        Map<String, Object> attributes = new HashMap<>();

        boolean accepted = interceptor.beforeHandshake(
                new ServletServerHttpRequest(servletRequest),
                new ServletServerHttpResponse(servletResponse),
                Mockito.mock(WebSocketHandler.class), attributes);

        assertTrue(accepted);
        assertEquals("7", attributes.get("userId"));
        assertEquals("ADMIN", attributes.get("userRole"));
    }

    @Test
    void rejectsHandshakeWithoutInternalToken() {
        MockHttpServletRequest servletRequest = request(null, "7", "ADMIN");
        MockHttpServletResponse servletResponse = new MockHttpServletResponse();

        boolean accepted = interceptor.beforeHandshake(
                new ServletServerHttpRequest(servletRequest),
                new ServletServerHttpResponse(servletResponse),
                Mockito.mock(WebSocketHandler.class), new HashMap<>());

        assertFalse(accepted);
        assertEquals(401, servletResponse.getStatus());
    }

    @Test
    void rejectsHandshakeWithForgedInternalToken() {
        MockHttpServletRequest servletRequest = request("forged:signature", "7", "ADMIN");
        MockHttpServletResponse servletResponse = new MockHttpServletResponse();

        boolean accepted = interceptor.beforeHandshake(
                new ServletServerHttpRequest(servletRequest),
                new ServletServerHttpResponse(servletResponse),
                Mockito.mock(WebSocketHandler.class), new HashMap<>());

        assertFalse(accepted);
        assertEquals(401, servletResponse.getStatus());
    }

    @Test
    void rejectsHandshakeWhenGatewayDidNotInjectUserId() {
        MockHttpServletRequest servletRequest = request(internalTokenUtil.generate(), null, "ADMIN");
        MockHttpServletResponse servletResponse = new MockHttpServletResponse();

        boolean accepted = interceptor.beforeHandshake(
                new ServletServerHttpRequest(servletRequest),
                new ServletServerHttpResponse(servletResponse),
                Mockito.mock(WebSocketHandler.class), new HashMap<>());

        assertFalse(accepted);
        assertEquals(401, servletResponse.getStatus());
    }

    private MockHttpServletRequest request(String internalToken, String userId, String userRole) {
        MockHttpServletRequest request = new MockHttpServletRequest();
        if (internalToken != null) {
            request.addHeader("X-Internal-Token", internalToken);
        }
        if (userId != null) {
            request.addHeader("X-User-Id", userId);
        }
        if (userRole != null) {
            request.addHeader("X-User-Role", userRole);
        }
        return request;
    }
}
