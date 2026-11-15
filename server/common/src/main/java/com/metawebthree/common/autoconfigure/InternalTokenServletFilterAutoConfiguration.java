package com.metawebthree.common.autoconfigure;

import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.autoconfigure.condition.ConditionalOnWebApplication;
import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.context.annotation.Bean;

import com.metawebthree.common.filter.InternalTokenFilter;
import com.metawebthree.common.utils.InternalTokenUtil;

import jakarta.servlet.Filter;

/**
 * Servlet-side enforcement of "gateway only" access.
 *
 * <p>The filter is registered explicitly through a {@link FilterRegistrationBean}
 * rather than relying on component scanning, and only for servlet applications.
 * The reactive gateway needs the token utility but must never install a servlet
 * filter. Services opt in with {@code internal-token.enabled=true}.</p>
 */
@AutoConfiguration(after = InternalTokenAutoConfiguration.class)
@ConditionalOnClass(Filter.class)
@ConditionalOnWebApplication(type = ConditionalOnWebApplication.Type.SERVLET)
@ConditionalOnProperty(prefix = "internal-token", name = "enabled", havingValue = "true")
public class InternalTokenServletFilterAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public FilterRegistrationBean<InternalTokenFilter> internalTokenFilter(
            InternalTokenUtil internalTokenUtil,
            InternalTokenProperties properties) {
        FilterRegistrationBean<InternalTokenFilter> registration = new FilterRegistrationBean<>(
                new InternalTokenFilter(internalTokenUtil, properties.getExcludedPaths()));
        registration.setName("internalTokenFilter");
        registration.addUrlPatterns("/*");
        registration.setOrder(-50);
        return registration;
    }
}
