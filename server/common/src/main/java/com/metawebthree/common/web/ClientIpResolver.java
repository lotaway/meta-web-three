package com.metawebthree.common.web;

import com.metawebthree.common.constants.HeaderConstants;

import jakarta.servlet.http.HttpServletRequest;

public final class ClientIpResolver {

    private ClientIpResolver() {
    }

    public static String resolve(HttpServletRequest request) {
        String trustedIp = request.getHeader(HeaderConstants.CLIENT_IP);
        if (trustedIp != null && !trustedIp.isBlank()) {
            return trustedIp.trim();
        }
        return request.getRemoteAddr();
    }
}