package com.metawebthree.common.filter;

import com.metawebthree.common.utils.InternalTokenUtil;
import jakarta.servlet.Filter;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.ServletRequest;
import jakarta.servlet.ServletResponse;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import com.metawebthree.common.utils.InternalTokenUtil;
import jakarta.servlet.Filter;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.ServletRequest;
import jakarta.servlet.ServletResponse;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.util.AntPathMatcher;

import java.io.IOException;
import java.util.List;

@Component
@Order(-50)
public class InternalTokenFilter implements Filter {

    private final InternalTokenUtil internalTokenUtil;
    private final List<String> excludedPaths;
    private final AntPathMatcher pathMatcher;

    public InternalTokenFilter(InternalTokenUtil internalTokenUtil,
                               @Value("${internal-token.excluded-paths:/actuator/**,/health,/info,/error,/swagger-ui/**,/v3/api-docs/**}") List<String> excludedPaths) {
        this.internalTokenUtil = internalTokenUtil;
        this.excludedPaths = excludedPaths;
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

        String token = httpRequest.getHeader("X-Internal-Token");
        if (token == null || !internalTokenUtil.validate(token)) {
            httpResponse.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            httpResponse.setContentType("application/json");
            httpResponse.getWriter().write("{\"code\":\"INTERNAL_TOKEN_INVALID\",\"message\":\"Invalid internal token\"}");
            return;
        }

        chain.doFilter(request, response);
    }

    private boolean shouldSkip(String path) {
        return excludedPaths.stream().anyMatch(pattern -> pathMatcher.match(pattern, path));
    }
}
