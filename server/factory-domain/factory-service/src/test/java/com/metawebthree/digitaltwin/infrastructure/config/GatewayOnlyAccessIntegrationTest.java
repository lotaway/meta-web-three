package com.metawebthree.digitaltwin.infrastructure.config;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.WebApplicationContextRunner;
import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import com.metawebthree.common.autoconfigure.InternalTokenAutoConfiguration;
import com.metawebthree.common.autoconfigure.InternalTokenServletFilterAutoConfiguration;
import com.metawebthree.common.filter.InternalTokenFilter;
import com.metawebthree.common.utils.InternalTokenUtil;

/**
 * HTTP-level integration test for the gateway-only boundary: the common
 * module's auto-configuration must actually register the filter, and the filter
 * must reject every non-excluded REST path without a gateway-signed token.
 */
class GatewayOnlyAccessIntegrationTest {

    private InternalTokenUtil internalTokenUtil;
    private MockMvc mockMvc;

    @BeforeEach
    void setUp() {
        internalTokenUtil = new InternalTokenUtil("integration-secret", 5);
        InternalTokenFilter filter = new InternalTokenFilter(
                internalTokenUtil,
                java.util.List.of("/actuator/**", "/health", "/info", "/error",
                        "/swagger-ui/**", "/v3/api-docs/**"));
        mockMvc = MockMvcBuilders.standaloneSetup(new ProbeController())
                .addFilters(filter)
                .build();
    }

    @Test
    void autoConfigurationRegistersTheFilterWhenEnabled() {
        new WebApplicationContextRunner()
                .withConfiguration(AutoConfigurations.of(
                        InternalTokenAutoConfiguration.class,
                        InternalTokenServletFilterAutoConfiguration.class))
                .withPropertyValues("internal-token.enabled=true")
                .run(context -> {
                    assertThat(context).hasSingleBean(FilterRegistrationBean.class);
                    FilterRegistrationBean<?> registration = context.getBean(FilterRegistrationBean.class);
                    assertThat(registration.getFilter()).isInstanceOf(InternalTokenFilter.class);
                    assertThat(registration.getOrder()).isEqualTo(-50);
                });
    }

    @Test
    void rejectsBusinessRequestWithoutInternalToken() throws Exception {
        mockMvc.perform(get("/api/digital-twin/devices"))
                .andExpect(status().isUnauthorized())
                .andExpect(jsonPath("$.code").value("INTERNAL_TOKEN_INVALID"));
    }

    @Test
    void rejectsForgedInternalToken() throws Exception {
        mockMvc.perform(get("/api/digital-twin/devices")
                        .header("X-Internal-Token", "forged:signature"))
                .andExpect(status().isUnauthorized());
    }

    @Test
    void acceptsRequestWithGatewaySignedToken() throws Exception {
        mockMvc.perform(get("/api/digital-twin/devices")
                        .header("X-Internal-Token", internalTokenUtil.generate()))
                .andExpect(status().isOk())
                .andExpect(content().string("devices"));
    }

    @Test
    void leavesHealthEndpointAvailableForContainerHealthChecks() throws Exception {
        mockMvc.perform(get("/actuator/health"))
                .andExpect(status().isOk())
                .andExpect(content().string("UP"));
    }

    @RestController
    static class ProbeController {
        @GetMapping("/api/digital-twin/devices")
        String devices() {
            return "devices";
        }

        @GetMapping("/actuator/health")
        String health() {
            return "UP";
        }
    }
}
