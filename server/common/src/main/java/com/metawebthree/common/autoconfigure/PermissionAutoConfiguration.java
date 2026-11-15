package com.metawebthree.common.autoconfigure;

import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnWebApplication;
import org.springframework.context.annotation.Bean;

import com.metawebthree.common.annotations.PermissionChecker;
import com.metawebthree.common.annotations.RequirePermissionAspect;
import com.metawebthree.common.auth.TokenBlacklistService;

/**
 * Activates {@code @RequirePermission} only in servlet services that actually
 * provide a {@link PermissionChecker}. Previously the aspect was a plain
 * {@code @Component} in the common jar, so it was invisible to applications
 * whose {@code scanBasePackages} did not include {@code common.annotations} and
 * the annotations were silently ignored.
 */
@AutoConfiguration
@ConditionalOnWebApplication(type = ConditionalOnWebApplication.Type.SERVLET)
@ConditionalOnBean(PermissionChecker.class)
public class PermissionAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public RequirePermissionAspect requirePermissionAspect(
            ObjectProvider<PermissionChecker> permissionCheckers,
            ObjectProvider<TokenBlacklistService> tokenBlacklistService) {
        return new RequirePermissionAspect(permissionCheckers, tokenBlacklistService);
    }
}
