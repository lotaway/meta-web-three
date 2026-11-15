package com.metawebthree.common.filter;

import java.io.IOException;
import java.util.List;

import org.springframework.util.AntPathMatcher;

import com.metawebthree.common.utils.InternalTokenUtil;

import jakarta.servlet.Filter;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.ServletRequest;
import jakarta.servlet.ServletResponse;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

/**
 * Rejects HTTP requests that did not pass through the gateway.
 *
 * <p>Registered by {@code InternalTokenServletFilterAutoConfiguration}; it is
 * not component-scanned, so enabling {@code internal-token.enabled} is the only
 * switch and it behaves identically in every servlet service.</p>
 */
public class InternalTokenFilter implements Filter {

    private static final String INTERNAL_TOKEN_HEADER = "X-Internal-Token";

    private final InternalTokenUtil internalTokenUtil;
    private final List<String> excludedPaths;
    private final AntPathMatcher pathMatcher;

    public InternalTokenFilter(InternalTokenUtil internalTokenUtil, List<String> excludedPaths) {
        this.internalTokenUtil = internalTokenUtil;
        this.excludedPaths = excludedPaths == null ? List.of() : excludedPaths;
        this.pathMatcher = new AntPathMatcher();
    }

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
            throws IOException, ServletException {
        HttpServletRequest httpRequest = (HttpServletRequest) request;
        HttpServletResponse httpResponse = (HttpServletResponse) response;

        if (shouldSkip(httpRequest.getRequestURI())) {
            chain.doFilter(request, response);
            return;
        }

        String token = httpRequest.getHeader(INTERNAL_TOKEN_HEADER);
        if (token == null || !internalTokenUtil.validate(token)) {
            httpResponse.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            httpResponse.setContentType("application/json");
            httpResponse.getWriter().write(
                    "{\"code\":\"INTERNAL_TOKEN_INVALID\",\"message\":\"Invalid internal token\"}");
            return;
        }

        chain.doFilter(request, response);
    }

    private boolean shouldSkip(String path) {
        return excludedPaths.stream().anyMatch(pattern -> pathMatcher.match(pattern, path));
    }
}
