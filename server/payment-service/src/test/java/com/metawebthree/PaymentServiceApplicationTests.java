package com.metawebthree;

import org.apache.dubbo.config.annotation.DubboReference;
import org.junit.Assert;
import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.test.context.junit4.SpringRunner;

import com.metawebthree.service.DecisionService;

@RunWith(SpringRunner.class)
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.DEFINED_PORT)
@ComponentScan(basePackages = {
        // "com.metawebthree",
        "com.metawebthree.common"
})
@EnableAutoConfiguration
class PaymentServiceApplicationTests {

    private DecisionService decisionService;

    @Test
    void testRPCService() {
        int score = decisionService.test();
        System.out.println(String.format("RPC Service test result: %s", score));
        Assert.assertTrue(score > 0);
    }

}