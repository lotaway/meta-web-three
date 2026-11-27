package com.metawebthree.wallet.infrastructure.solana;

import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.autoconfigure.context.ConfigurationPropertiesAutoConfiguration;
import org.springframework.boot.test.context.ConfigDataApplicationContextInitializer;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;

class SolanaPropertiesTest {

    private final ApplicationContextRunner runner = new ApplicationContextRunner()
        .withInitializer(new ConfigDataApplicationContextInitializer())
        .withConfiguration(AutoConfigurations.of(ConfigurationPropertiesAutoConfiguration.class))
        .withUserConfiguration(SolanaProperties.class);

    @Test
    void bindsAllExternalAddressesFromApplicationYml() {
        runner.run(context -> {
            SolanaProperties props = context.getBean(SolanaProperties.class);
            assertThat(props.getProgramId())
                .isEqualTo("FiSvBZn8sSV3r8xC6PcEC59D8oBeDUj2sdS9Jdppq9oW");
            assertThat(props.getPaymentMint())
                .isEqualTo("So11111111111111111111111111111111111111112");
            assertThat(props.getPrograms().getToken())
                .isEqualTo("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA");
            assertThat(props.getPrograms().getAssociatedToken())
                .isEqualTo("ATokenGPvbdGVxr1b2hvZbsiqW5xr25ix9f2JtAbjv9w");
            assertThat(props.getPrograms().getMetadata())
                .isEqualTo("metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s");
            assertThat(props.getPrograms().getSystem())
                .isEqualTo("11111111111111111111111111111111");
            assertThat(props.getPrograms().getRent())
                .isEqualTo("SysvarRent111111111111111111111111111111111");
            assertThat(props.getEncryptionKey()).isEmpty();
            assertThat(props.getDefaultRewardPercentages())
                .containsExactly(5000, 3000, 2000);
        });
    }

    @Test
    void externalAddressesCanBeOverridden() {
        runner
            .withSystemProperties(
                "SOLANA_PROGRAM_ID=11111111111111111111111111111111",
                "SOLANA_PAYMENT_MINT=So11111111111111111111111111111111111111113",
                "SOLANA_ACTIVITY_DEFAULT_REWARD_PERCENTAGES=1000,2000,7000")
            .run(context -> {
                SolanaProperties props = context.getBean(SolanaProperties.class);
                assertThat(props.getProgramId()).isEqualTo("11111111111111111111111111111111");
                assertThat(props.getPaymentMint())
                    .isEqualTo("So11111111111111111111111111111111111111113");
                assertThat(props.getDefaultRewardPercentages())
                    .containsExactly(1000, 2000, 7000);
                // Unspecified values keep their yml defaults.
                assertThat(props.getPrograms().getAssociatedToken())
                    .isEqualTo("ATokenGPvbdGVxr1b2hvZbsiqW5xr25ix9f2JtAbjv9w");
            });
    }
}
