package com.metawebthree;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.TestPropertySource;

/**
 * Test class for Quartz configuration
 */
@SpringBootTest
@TestPropertySource(properties = {
    "spring.quartz.job-store-type=memory",
    "spring.quartz.auto-startup=false"
})
class QuartzConfigTest {

    @Test
    void contextLoads() {
        // This test will verify that the Spring context loads successfully
        // with Quartz configuration
    }
}
