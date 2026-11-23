package com.metawebthree.common.autoconfigure;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.boot.test.context.runner.ReactiveWebApplicationContextRunner;
import org.springframework.boot.test.context.runner.WebApplicationContextRunner;
import org.springframework.boot.web.servlet.FilterRegistrationBean;

import com.metawebthree.common.annotations.PermissionChecker;
import com.metawebthree.common.annotations.RequirePermissionAspect;
import com.metawebthree.common.event.DomainEventPublisher;
import com.metawebthree.common.filter.InternalTokenFilter;
import com.metawebthree.common.utils.InternalTokenUtil;

class InternalTokenAutoConfigurationTest {

    private static final AutoConfigurations CONFIGURATIONS = AutoConfigurations.of(
            InternalTokenAutoConfiguration.class,
            InternalTokenServletFilterAutoConfiguration.class,
            PermissionAutoConfiguration.class,
            DomainEventAutoConfiguration.class);

    @Test
    void registersServletFilterWhenEnabled() {
        new WebApplicationContextRunner()
                .withConfiguration(CONFIGURATIONS)
                .withPropertyValues("internal-token.enabled=true")
                .run(context -> {
                    assertThat(context).hasSingleBean(InternalTokenUtil.class);
                    assertThat(context).hasSingleBean(FilterRegistrationBean.class);
                    FilterRegistrationBean<?> registration = context.getBean(FilterRegistrationBean.class);
                    assertThat(registration.getFilter()).isInstanceOf(InternalTokenFilter.class);
                    assertThat(registration.getOrder()).isEqualTo(-50);
                });
    }

    @Test
    void doesNotRegisterServletFilterUnlessOptedIn() {
        new WebApplicationContextRunner()
                .withConfiguration(CONFIGURATIONS)
                .run(context -> {
                    assertThat(context).hasSingleBean(InternalTokenUtil.class);
                    assertThat(context).doesNotHaveBean(FilterRegistrationBean.class);
                });
    }

    @Test
    void reactiveGatewayGetsOnlyTheTokenUtility() {
        new ReactiveWebApplicationContextRunner()
                .withConfiguration(CONFIGURATIONS)
                .withPropertyValues("internal-token.enabled=true")
                .run(context -> {
                    assertThat(context).hasSingleBean(InternalTokenUtil.class);
                    assertThat(context).doesNotHaveBean(FilterRegistrationBean.class);
                });
    }

    @Test
    void bindsSecretAndWindowFromProperties() {
        new ApplicationContextRunner()
                .withConfiguration(AutoConfigurations.of(InternalTokenAutoConfiguration.class))
                .withPropertyValues(
                        "internal-token.secret=custom-secret",
                        "internal-token.timeout-seconds=9")
                .run(context -> {
                    InternalTokenProperties properties = context.getBean(InternalTokenProperties.class);
                    assertThat(properties.getSecret()).isEqualTo("custom-secret");
                    assertThat(properties.getTimeoutSeconds()).isEqualTo(9);
                });
    }

    @Test
    void blankSecretStillCreatesUtilityWithRandomFallback() {
        new ApplicationContextRunner()
                .withConfiguration(AutoConfigurations.of(InternalTokenAutoConfiguration.class))
                .run(context -> {
                    assertThat(context).hasSingleBean(InternalTokenUtil.class);
                    // Fail-closed: a random secret is generated, no public default baked in.
                    assertThat(context.getBean(InternalTokenProperties.class).getSecret()).isNull();
                });
    }

    @Test
    void configuredSecretSignsAndValidates() {
        new ApplicationContextRunner()
                .withConfiguration(AutoConfigurations.of(InternalTokenAutoConfiguration.class))
                .withPropertyValues("internal-token.secret=real-configured-secret")
                .run(context -> {
                    InternalTokenUtil util = context.getBean(InternalTokenUtil.class);
                    String token = util.generate();
                    assertThat(util.validate(token)).isTrue();
                    assertThat(util.validate(token + "tampered")).isFalse();
                });
    }

    @Test
    void permissionAspectIsCreatedOnlyWhenACheckerExists() {
        new WebApplicationContextRunner()
                .withConfiguration(AutoConfigurations.of(PermissionAutoConfiguration.class))
                .run(context -> assertThat(context).doesNotHaveBean(RequirePermissionAspect.class));

        new WebApplicationContextRunner()
                .withConfiguration(AutoConfigurations.of(PermissionAutoConfiguration.class))
                .withBean(PermissionChecker.class, () -> (userId, permission) -> true)
                .run(context -> assertThat(context).hasSingleBean(RequirePermissionAspect.class));
    }

    @Test
    void kafkaPublisherIsCreatedOnlyWhenKafkaTemplateExists() {
        new ApplicationContextRunner()
                .withConfiguration(AutoConfigurations.of(DomainEventAutoConfiguration.class))
                .run(context -> assertThat(context).doesNotHaveBean(DomainEventPublisher.class));
    }
}
