package com.metawebthree.wallet.interfaces.controller;

import com.metawebthree.wallet.application.service.SolanaActivityService;
import com.metawebthree.wallet.application.service.SolanaCommissionService;
import com.metawebthree.wallet.application.service.SolanaCouponService;
import com.metawebthree.wallet.application.service.SolanaMarketplaceService;
import com.metawebthree.wallet.application.service.SolanaTokenService;
import com.metawebthree.wallet.infrastructure.solana.SolanaWalletManager;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.ConfigDataApplicationContextInitializer;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

class SolanaEnabledTest {

    /**
     * Registers the real controller classes so their class-level
     * {@code @SolanaEnabled} (a meta-annotation for {@code @ConditionalOnProperty})
     * is evaluated by Spring.
     */
    private ApplicationContextRunner runner() {
        return new ApplicationContextRunner()
            .withUserConfiguration(
                SolanaTokenController.class,
                SolanaWalletController.class,
                SolanaMarketplaceController.class,
                SolanaCouponController.class,
                SolanaActivityController.class,
                SolanaCommissionController.class)
            .withBean(SolanaTokenService.class, () -> mock(SolanaTokenService.class))
            .withBean(SolanaWalletManager.class, () -> mock(SolanaWalletManager.class))
            .withBean(SolanaMarketplaceService.class, () -> mock(SolanaMarketplaceService.class))
            .withBean(SolanaCouponService.class, () -> mock(SolanaCouponService.class))
            .withBean(SolanaActivityService.class, () -> mock(SolanaActivityService.class))
            .withBean(SolanaCommissionService.class, () -> mock(SolanaCommissionService.class));
    }

    @Test
    void applicationYml_defaultsSwitchOffAndParses() {
        // Loads the real application.yml: guards against duplicate YAML keys and
        // verifies the opt-in switch defaults to false without env overrides.
        new ApplicationContextRunner()
            .withInitializer(new ConfigDataApplicationContextInitializer())
            .withUserConfiguration(SolanaTokenController.class)
            .withBean(SolanaTokenService.class, () -> mock(SolanaTokenService.class))
            .run(context -> {
                assertThat(context.getEnvironment()
                    .getProperty("blockchain.solana.enabled", Boolean.class)).isFalse();
                assertThat(context).doesNotHaveBean(SolanaTokenController.class);
            });
    }

    @Test
    void disabledByDefault_noSolanaControllerBean() {
        runner().run(context ->
            assertThat(context).doesNotHaveBean(SolanaTokenController.class));
    }

    @Test
    void explicitlyDisabled_noSolanaControllerBean() {
        runner()
            .withPropertyValues("blockchain.solana.enabled=false")
            .run(context -> {
                assertThat(context).doesNotHaveBean(SolanaTokenController.class);
                assertThat(context).doesNotHaveBean(SolanaCommissionController.class);
                assertThat(context).doesNotHaveBean(SolanaActivityController.class);
                assertThat(context).doesNotHaveBean(SolanaMarketplaceController.class);
                assertThat(context).doesNotHaveBean(SolanaCouponController.class);
                assertThat(context).doesNotHaveBean(SolanaWalletController.class);
            });
    }

    @Test
    void enabled_allSolanaControllerBeansPresent() {
        runner()
            .withPropertyValues("blockchain.solana.enabled=true")
            .run(context -> {
                assertThat(context).hasSingleBean(SolanaTokenController.class);
                assertThat(context).hasSingleBean(SolanaCommissionController.class);
                assertThat(context).hasSingleBean(SolanaActivityController.class);
                assertThat(context).hasSingleBean(SolanaMarketplaceController.class);
                assertThat(context).hasSingleBean(SolanaCouponController.class);
                assertThat(context).hasSingleBean(SolanaWalletController.class);
            });
    }
}
