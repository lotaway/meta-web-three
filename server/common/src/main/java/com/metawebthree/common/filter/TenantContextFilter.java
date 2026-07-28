package com.metawebthree.common.filter;

import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.common.context.TenantContext;
import jakarta.servlet.Filter;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.ServletRequest;
import jakarta.servlet.ServletResponse;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;

import java.io.IOException;

@Component
@Order(-40)
public class TenantContextFilter implements Filter {

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
            throws IOException, ServletException {
        HttpServletRequest httpRequest = (HttpServletRequest) request;
        try {
            String tenantIdStr = httpRequest.getHeader(HeaderConstants.TENANT_ID);
            if (StringUtils.hasText(tenantIdStr)) {
                TenantContext.setTenantId(Long.valueOf(tenantIdStr));
            }
            chain.doFilter(request, response);
        } finally {
            TenantContext.clear();
        }
    }
}
