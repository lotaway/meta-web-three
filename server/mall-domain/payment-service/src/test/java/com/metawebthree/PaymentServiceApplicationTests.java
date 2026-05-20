package com.metawebthree;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.test.context.junit4.SpringRunner;

import com.metawebthree.payment.application.DecisionService;

import org.junit.Assert;

@RunWith(SpringRunner.class)
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.DEFINED_PORT)
@ComponentScan(basePackages = {
        "com.metawebthree"
})
@EnableAutoConfiguration
class PaymentServiceApplicationTests {

    @Autowired
    private DecisionService decisionService;

    // @Test
    // @Disabled("future feature test")
    // void testRPCService() {
    //     int score = decisionService.test();
    //     System.out.println(String.format("RPC Service test result: %s", score));
    //     Assert.assertTrue(score > 0);
    // }

}